"""Fresh, matched-budget benchmark. Never imports or overwrites historical results."""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import shlex
import subprocess
import sys

import torch
from transformers import AutoTokenizer
from transformers.utils.hub import cached_file

from src.baseline import UncompressedBaselineEngine
from src.streaming_cache import StreamingLLMEngine
from src.sliding_cache import PureSlidingWindowEngine
from src.h2o_cache import H2OCacheEngine
from src.ogc_cache import OGCCacheEngine
from src.generation import DEFAULT_MODEL, DEFAULT_REVISION
from src.metrics import checkpoint_device_mib
from src.research import build_cases, score, summarize

POLICIES = ("full", "streaming", "sliding", "h2o_prefill_approx", "ogc", "query_tokens",
            "random_chunks", "adaptive_chunks", "entity_chunks")
AVAILABLE_POLICIES = POLICIES + ("ogc_corrected", "entity_balanced", "ogc_c32", "ogc_history")


def make_engines(model, revision, device, dtype, budgets, policies, chunk_size, prefill_chunk_size, seed,
                 modified_history_fraction=0.5):
    common = dict(model_name_or_path=model, revision=revision, device=device, dtype=dtype,
                  attn_implementation="eager")
    engines = {}
    if "full" in policies:
        engines["full"] = UncompressedBaselineEngine(**common)
    for budget in budgets:
        if budget < 68 or chunk_size < 1:
            raise ValueError("Budget must be >=68 and chunk size positive")
        # 1/4 of non-sink space for history, remainder for recent tokens.
        # B=260 -> S=4, H=64, W=192. Every policy gets exactly B total slots.
        history = (budget - 4) // 4
        for policy in policies:
            if policy == "full":
                continue
            if policy == "streaming":
                engine = StreamingLLMEngine(**common, sink_tokens=4, window_tokens=budget - 4)
            elif policy == "sliding":
                engine = PureSlidingWindowEngine(**common, window_tokens=budget)
            elif policy == "h2o_prefill_approx":
                engine = H2OCacheEngine(**common, heavy_hitters=history, recent_tokens=budget - history)
            else:
                kept_history = int((budget-4)*modified_history_fraction) if policy in {"entity_balanced", "ogc_history"} else history
                actual_chunk = 32 if policy == "ogc_c32" else chunk_size
                # Fixed chunk grids (16/32/64) divide H at all matched budgets.
                if kept_history % actual_chunk:
                    raise ValueError("Chunk size must divide historical capacity")
                engine = OGCCacheEngine(model_name=model, revision=revision, device=device, dtype=dtype,
                                        sink_tokens=4, recent_window=budget - 4 - kept_history,
                                        chunk_size=actual_chunk, num_chunks=kept_history // actual_chunk,
                                        selection_policy="ogc" if policy in {"ogc_c32", "ogc_history"} else policy, seed=seed)
            name = f"{policy}/B{budget}"
            if policy in {"ogc", "random_chunks", "ogc_corrected", "ogc_c32", "ogc_history"}:
                name += f"/C{32 if policy == 'ogc_c32' else chunk_size}"
            if policy in {"entity_balanced", "ogc_history"}:
                name += f"/H{kept_history}"
            engines[name] = engine
    for engine in engines.values():
        engine.prefill_chunk_size = prefill_chunk_size
    return engines


def prepare_item(item, tokenizer, protocol):
    if protocol == "held_query":
        position = item["document"].rfind("[Inquiry]")
        if position < 0:
            raise ValueError("Held-query evaluation requires an explicit [Inquiry] boundary")
        offsets = tokenizer(item["document"], return_offsets_mapping=True)["offset_mapping"]
        item["document_token_count"] = next(i for i, (_, end) in enumerate(offsets) if end > position)
        if not 0 < item["document_token_count"] < len(offsets):
            raise ValueError("Empty document or question at held-query boundary")
    return item


def configure_item(engine, item, protocol):
    engine.document_token_count = item.get("document_token_count") if protocol == "held_query" else None
    engine.selection_query = item["question"]


def record_result(result, item, config):
    raw = asdict(result)
    prediction = raw.get("output_text", raw.get("generated_text"))
    physical = raw.get("compressed_kv_bytes", raw.get("kv_cache_bytes"))
    checkpoints = raw.get("checkpoints", raw.get("memory_metrics", {}))
    row = dict(item_id=item["item_id"], config_name=config,
               context_length=item["target_context_length"], actual_input_tokens=result.input_tokens,
               depth_category=item["depth_category"], query_type=item["query_type"],
               ground_truths=item["ground_truth_answers"], prediction=prediction,
               compressed_kv_bytes=physical, cache_storage_bytes=result.cache_storage_bytes,
               uncompressed_counterfactual_kv_bytes=raw.get("uncompressed_kv_bytes", raw.get("theoretical_kv_bytes")),
               prefill_kv_bytes=result.prefill_kv_bytes,
               actual_cached_tokens=len(result.retained_positions),
               generated_tokens=len(result.generated_token_ids),
               selected_chunk_ranges=raw.get("selected_chunk_ranges"),
               prefill_time_ms=result.prefill_time_ms, scoring_time_ms=result.scoring_time_ms,
               prefill_forward_time_ms=result.prefill_forward_time_ms,
               selection_time_ms=result.selection_time_ms, initial_selection_time_ms=result.initial_selection_time_ms,
               decode_pruning_time_ms=result.decode_pruning_time_ms, decode_time_ms=result.decode_time_ms,
               decode_forward_time_ms=result.decode_forward_time_ms, decode_steps=result.decode_steps,
               ttft_ms=result.ttft_ms, total_time_ms=result.total_time_ms,
               checkpoint_max_device_allocated_mib=checkpoint_device_mib(checkpoints),
               cuda_peak_allocated_bytes=result.cuda_peak_allocated_bytes,
               generated_token_ids=result.generated_token_ids,
               retained_positions=result.retained_positions, cache_lengths=result.cache_lengths,
               memory_scope=result.memory_scope, cache_position_mode=result.cache_position_mode,
               checkpoints=checkpoints)
    for name in ("evaluation_protocol", "document_tokens", "query_tokens", "query_time_ms",
                 "query_forward_time_ms", "query_pruning_time_ms", "query_cache_lengths",
                 "document_retained_positions"):
        row[name] = getattr(result, name)
    row.update(score(prediction, item["ground_truth_answers"]))
    return row


def main(default_policies=POLICIES):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--budgets", type=int, nargs="+", default=[260, 516, 1028])
    parser.add_argument("--lengths", type=int, nargs="+", default=[1024, 2048, 4096, 8192])
    parser.add_argument("--policies", nargs="+", choices=AVAILABLE_POLICIES, default=default_policies)
    parser.add_argument("--protocol", choices=["continuation", "held_query"], default="continuation")
    parser.add_argument("--modified-history-fraction", type=float, choices=[0.25, 0.5, 0.75], default=0.5)
    parser.add_argument("--repeats", type=int, default=5, help="Independent cases per length/depth/type cell; default 300 total")
    parser.add_argument("--seed", type=int, default=20261031)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--prefill-chunk-size", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=35)
    parser.add_argument("--dataset", help="Optional existing dataset; otherwise build independent research cases")
    parser.add_argument("--max-items", type=int, help="Seeded random pilot subset; explicitly recorded")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--output", required=True, help="New output directory; existing paths are rejected")
    args = parser.parse_args()
    if args.repeats < 1 or args.max_new_tokens < 1 or args.prefill_chunk_size < 1 or (args.max_items is not None and args.max_items < 1):
        parser.error("Repeats, output tokens, block size and optional item cap must be positive")
    if args.model != DEFAULT_MODEL and args.revision == DEFAULT_REVISION:
        parser.error("A different model requires its own explicit pinned --revision")
    if len(args.revision) != 40 or any(c not in "0123456789abcdef" for c in args.revision):
        parser.error("--revision must be a full immutable commit hash")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    # All input/output tensors are initialized by their producing operations.
    # Avoid NaN-filling every temporary allocation on MPS under deterministic mode.
    torch.utils.deterministic.fill_uninitialized_memory = False
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, use_fast=True)
    items = json.loads(Path(args.dataset).read_text()) if args.dataset else build_cases(tokenizer, args.lengths, args.repeats, args.seed)
    # Interleave independent repetitions; small pilots use a recorded seeded sample.
    groups = {}
    for item in items:
        key = (item["target_context_length"], item["depth_category"], item["query_type"])
        groups.setdefault(key, []).append(item)
    items = [group[i] for i in range(max(map(len, groups.values()))) for group in groups.values() if i < len(group)]
    if args.max_items:
        # Seeded shuffle samples across lengths for very small pilots as well.
        random.Random(args.seed).shuffle(items)
        items = items[:args.max_items]
    if len({item["item_id"] for item in items}) != len(items):
        raise ValueError("Dataset contains duplicate question ids")
    items = [prepare_item(item, tokenizer, args.protocol) for item in items]
    dataset_text = json.dumps(items, indent=2)
    (out / "dataset.json").write_text(dataset_text)
    files = [Path("src") / f"{name}.py" for name in
             ("baseline", "streaming_cache", "sliding_cache", "h2o_cache", "ogc_cache",
              "generation", "selection", "metrics", "dataset", "research")]
    files.append(Path(__file__))
    manifest = dict(config=vars(args), started_at=datetime.now().astimezone().isoformat(),
                    command=shlex.join([sys.executable, *sys.argv]),
                    git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                    git_status=subprocess.check_output(["git", "status", "--porcelain"], text=True),
                    source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                    dataset_sha256=hashlib.sha256(dataset_text.encode()).hexdigest(),
                    python=sys.version, platform=platform.platform(), machine=platform.machine(),
                    versions={p: importlib.metadata.version(p) for p in ["torch", "transformers", "numpy", "tokenizers", "safetensors"]},
                    device=args.device, attention_backend="eager", cases=len(items),
                    prompt_format="dataset supplied verbatim" if args.dataset else "tokenizer chat template, single user message",
                    generation_settings=dict(do_sample=False, temperature=0.0, repetition_penalty=1.0,
                                             stop="model/tokenizer EOS only; no implicit newline stop"),
                    determinism=dict(algorithms=True, warn_only=True, fill_uninitialized_memory=False),
                    prefill_scope="full prompt KV; block attention; post-prefill pruning",
                    memory_note="MPS allocations are checkpoints, RSS peak is process lifetime; neither is a per-run peak. CUDA peak reset per question.",
                    first_token_note=("Document compressed before any question token is forwarded; token 1 uses post-query logits. Query tokens count within B; selector may read question text lexically."
                                      if args.protocol == "held_query" else "Token 1 uses full-prefill logits; historical continuation protocol."),
                    warmup="one 4-token generation per config on shortest evaluated document, excluded from results",
                    status="dataset_only" if args.build_only else "running")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    if args.build_only:
        print(f"Built {len(items)} cases in {out}; no inference performed", flush=True)
        return
    engines = make_engines(args.model, args.revision, args.device, getattr(torch, args.dtype),
                           args.budgets, args.policies, args.chunk_size, args.prefill_chunk_size, args.seed,
                           args.modified_history_fraction)
    # Share identical weights/tokenizer. Every policy reruns its own prefill and decode.
    owner = next(iter(engines.values()))
    try:
        owner.load_model()
        if str(owner.device) != args.device:
            raise RuntimeError("Actual device differs from requested device")
        config_path = Path(cached_file(args.model, "config.json", revision=args.revision))
        resolved_revision = config_path.parent.name
        if resolved_revision != args.revision:
            raise RuntimeError(f"Resolved cached snapshot differs: {resolved_revision}")
        manifest["resolved_revision"] = resolved_revision
        manifest["model_config_sha256"] = hashlib.sha256(config_path.read_bytes()).hexdigest()
        warm_item = min(items, key=lambda item: item["actual_token_length"])
        for engine in engines.values():
            engine.model, engine.tokenizer = owner.model, tokenizer
            engine.max_context_length = owner.max_context_length
            configure_item(engine, warm_item, args.protocol)
            engine.generate(warm_item["document"], max_new_tokens=4)
        print(f"Warmup complete: {len(engines)} configurations", flush=True)
    except BaseException as exc:
        manifest.update(status="failed_before_measurement", error=f"{type(exc).__name__}: {exc}",
                        finished_at=datetime.now().astimezone().isoformat())
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
        raise
    records = []
    order_rng = random.Random(args.seed)
    try:
        with (out / "outputs.jsonl").open("x") as handle:
            for index, item in enumerate(items):
                order = list(engines)
                order_rng.shuffle(order)
                for config in order:
                    engine = engines[config]
                    # Random-chunk draws are independent per document, reproducible across budgets.
                    if hasattr(engine, "seed"):
                        engine.seed = item["seed"] ^ args.seed
                    configure_item(engine, item, args.protocol)
                    result = engine.generate(item["document"], args.max_new_tokens, do_sample=False)
                    row = record_result(result, item, config)
                    row["execution_index"] = len(records)
                    row["case_seed"] = item["seed"]
                    handle.write(json.dumps(row) + "\n")
                    handle.flush()
                    records.append(row)
                print(f"[{index+1}/{len(items)}] {len(records)} outputs persisted; L={item['actual_token_length']}", flush=True)
                first_tokens = {row["generated_token_ids"][0] for row in records if row["item_id"] == item["item_id"]}
                if args.protocol == "continuation" and len(first_tokens) != 1:
                    raise AssertionError("Identical prefill produced different first tokens across policies")
        (out / "summary.json").write_text(json.dumps(summarize(records), indent=2))
        manifest["status"] = "completed"
        manifest["outputs"] = len(records)
    except BaseException as exc:
        manifest["status"] = "failed_or_interrupted"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        manifest["finished_at"] = datetime.now().astimezone().isoformat()
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
