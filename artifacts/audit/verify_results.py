"""Read-only validation of completed benchmark artifacts; writes a separate audit."""
import hashlib
import json
from pathlib import Path
import re
import sys
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from transformers import AutoTokenizer
from src.research import score


def verify(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["status"] == "completed", manifest["status"]
    dataset = json.loads((directory / "dataset.json").read_text())
    assert hashlib.sha256((directory / "dataset.json").read_bytes()).hexdigest() == manifest["dataset_sha256"]
    items = {item["item_id"]: item for item in dataset}
    assert len(items) == len(dataset) == manifest["cases"]
    records = [json.loads(line) for line in (directory / "outputs.jsonl").read_text().splitlines()]
    summary = json.loads((directory / "summary.json").read_text())
    config = manifest["config"]
    assert config["model"] == "Qwen/Qwen2.5-1.5B-Instruct"
    bytes_per_token = 2 * 28 * 2 * 128 * {"float16": 2, "bfloat16": 2, "float32": 4}[config["dtype"]]
    expected_configs = set()
    for policy in config["policies"]:
        if policy == "full":
            expected_configs.add("full")
        else:
            for budget in config["budgets"]:
                name = f"{policy}/B{budget}"
                if policy in {"ogc", "random_chunks"}:
                    name += f"/C{config['chunk_size']}"
                expected_configs.add(name)
    assert set(summary) == expected_configs
    assert len(records) == len(items) * len(expected_configs) == manifest["outputs"]
    assert [r["execution_index"] for r in records] == list(range(len(records)))
    tokenizer = AutoTokenizer.from_pretrained(config["model"], revision=config["revision"], local_files_only=True)
    by_item = defaultdict(list)
    for row in records:
        item = items[row["item_id"]]
        n = len(tokenizer.encode(item["document"]))
        assert n == row["actual_input_tokens"] == item["actual_token_length"]
        positions, steps = row["retained_positions"], row["decode_steps"]
        match = re.search(r"/B(\d+)", row["config_name"])
        budget = int(match.group(1)) if match else n + steps
        assert positions == sorted(set(positions))
        assert positions[-1] == n + steps - 1
        assert row["cache_lengths"] == [min(n + step, budget) for step in range(steps + 1)]
        assert len(positions) == row["actual_cached_tokens"] == min(n + steps, budget)
        assert row["compressed_kv_bytes"] == row["cache_storage_bytes"] == len(positions) * bytes_per_token
        assert row["prefill_kv_bytes"] == n * bytes_per_token
        assert row["uncompressed_counterfactual_kv_bytes"] == (n + steps) * bytes_per_token
        assert len(row["generated_token_ids"]) == row["generated_tokens"] == steps + 1
        assert tokenizer.decode(row["generated_token_ids"], skip_special_tokens=True) == row["prediction"]
        assert row["ground_truths"] == item["ground_truth_answers"]
        expected_score = score(row["prediction"], row["ground_truths"])
        assert all(row[key] == value for key, value in expected_score.items())
        assert row["total_time_ms"] >= row["ttft_ms"] + row["decode_time_ms"]
        assert row["ttft_ms"] >= row["prefill_time_ms"] + row["initial_selection_time_ms"]
        assert row["decode_time_ms"] >= row["decode_forward_time_ms"] + row["decode_pruning_time_ms"]
        assert abs(row["selection_time_ms"] - row["initial_selection_time_ms"] - row["decode_pruning_time_ms"]) < 1e-6
        by_item[row["item_id"]].append(row)
    for rows in by_item.values():
        assert {row["config_name"] for row in rows} == expected_configs
        assert len({row["generated_token_ids"][0] for row in rows}) == 1
    for name, result in summary.items():
        rows = [r for r in records if r["config_name"] == name]
        assert result["total"] == len(rows)
        assert abs(result["accuracy_pct"] - 100 * sum(r["is_correct"] for r in rows) / len(rows)) < 1e-9
    result = dict(directory=str(directory), cases=len(items), outputs=len(records), configurations=len(expected_configs),
                  bytes_per_token=bytes_per_token, independently_seeded_cases=len({i["seed"] for i in dataset}),
                  checks="dataset hash, exact question/config pairing, raw output decoding/scoring, per-step budgets, absolute positions, physical bytes vs independent model geometry, timing decomposition, first-token agreement, summary accuracy",
                  passed=True)
    (directory / "artifact-verification.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    for directory in sys.argv[1:]:
        verify(directory)
