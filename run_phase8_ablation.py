"""
Phase 8 Ablation Study Runner:
Evaluates Observation-Guided Chunked Cache (OGC-Cache) against:
  1. Full Baseline (Uncompressed)
  2. StreamingLLM (S=4, W=256, Budget=260)
  3. Pure Sliding Window (W=260, Budget=260)
  4. H2O Heavy-Hitter (H=64, W=196, Budget=260)
  5. OGC-Cache (K=1, C=64, Budget=260)
  6. OGC-Cache (K=2, C=32, Budget=260)
"""

import gc
import json
import os
import sys
import time
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.ogc_cache import OGCCacheEngine


def load_dataset(filepath: str = "data/benchmark_dataset.json") -> List[Dict[str, Any]]:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_ogc_config(
    engine: OGCCacheEngine,
    dataset: List[Dict[str, Any]],
    config_name: str,
) -> Dict[str, Any]:
    print(f"\n{'='*75}")
    print(f"[Phase 8] Evaluating {config_name} across {len(dataset)} items...")
    print(f"{'='*75}")

    items_results = []
    total_correct = 0

    by_depth: Dict[str, Dict[str, int]] = {
        "start": {"correct": 0, "total": 0},
        "early_middle": {"correct": 0, "total": 0},
        "middle": {"correct": 0, "total": 0},
        "late_middle": {"correct": 0, "total": 0},
        "end": {"correct": 0, "total": 0},
    }

    by_context: Dict[int, Dict[str, Any]] = {
        1024: {"correct": 0, "total": 0, "prefill_ms": [], "decode_tps": [], "savings": []},
        2048: {"correct": 0, "total": 0, "prefill_ms": [], "decode_tps": [], "savings": []},
        4096: {"correct": 0, "total": 0, "prefill_ms": [], "decode_tps": [], "savings": []},
    }

    for idx, item in enumerate(dataset, 1):
        item_id = item["item_id"]
        ctx_len = item["target_context_length"]
        depth = item["depth_category"]
        prompt = item["document"]
        gts = item["ground_truth_answers"]

        print(
            f"  [{idx:2d}/{len(dataset)}] {config_name} | {item_id} "
            f"(L={item['actual_token_length']}, Depth={depth})... ",
            end="",
            flush=True,
        )

        res = engine.generate(prompt=prompt, max_new_tokens=35)
        pred = res.generated_text

        is_correct = all(gt.lower() in pred.lower() for gt in gts)
        if is_correct:
            total_correct += 1
            status_str = "[PASS]"
        else:
            status_str = "[FAIL]"

        by_depth[depth]["total"] += 1
        if is_correct:
            by_depth[depth]["correct"] += 1

        by_context[ctx_len]["total"] += 1
        if is_correct:
            by_context[ctx_len]["correct"] += 1
        by_context[ctx_len]["prefill_ms"].append(res.prefill_time_ms)
        by_context[ctx_len]["decode_tps"].append(res.decode_tokens_per_sec)
        by_context[ctx_len]["savings"].append(res.kv_cache_savings_pct)

        print(
            f"{status_str} KV: {res.compressed_kv_bytes / (1024**2):.2f} MiB "
            f"({res.kv_cache_savings_pct:.1f}% sav) | "
            f"Prefill: {res.prefill_time_ms:.1f}ms | Dec: {res.decode_tokens_per_sec:.1f} tok/s"
        )

        items_results.append({
            "item_id": item_id,
            "context_length": ctx_len,
            "depth_category": depth,
            "depth_fraction": item["depth_fraction"],
            "query_type": item["query_type"],
            "ground_truths": gts,
            "prediction": pred,
            "is_correct": is_correct,
            "prefill_time_ms": res.prefill_time_ms,
            "decode_tokens_per_sec": res.decode_tokens_per_sec,
            "uncompressed_kv_bytes": res.uncompressed_kv_bytes,
            "compressed_kv_bytes": res.compressed_kv_bytes,
            "kv_cache_savings_pct": res.kv_cache_savings_pct,
            "selected_chunk_ranges": res.selected_chunk_ranges,
            "peak_device_allocated_mib": res.memory_metrics.get("peak_device_allocated_mib", 0.0),
        })

    total_eval = len(dataset)
    acc_pct = (total_correct / total_eval) * 100.0 if total_eval > 0 else 0.0

    all_prefill = [r["prefill_time_ms"] for r in items_results]
    all_decode = [r["decode_tokens_per_sec"] for r in items_results]
    all_savings = [r["kv_cache_savings_pct"] for r in items_results]

    summary_depth = {}
    for d, info in by_depth.items():
        tot = info["total"]
        cor = info["correct"]
        summary_depth[d] = {
            "correct": cor,
            "total": tot,
            "accuracy_pct": (cor / tot * 100.0) if tot > 0 else 0.0,
        }

    summary_ctx = {}
    for c, info in by_context.items():
        tot = info["total"]
        cor = info["correct"]
        summary_ctx[str(c)] = {
            "correct": cor,
            "total": tot,
            "accuracy_pct": (cor / tot * 100.0) if tot > 0 else 0.0,
            "kv_savings_pct": float(np.mean(info["savings"])),
            "avg_prefill_ms": float(np.mean(info["prefill_ms"])),
            "avg_decode_tok_s": float(np.mean(info["decode_tps"])),
        }

    return {
        "config_name": config_name,
        "total_evaluated": total_eval,
        "total_correct": total_correct,
        "overall_accuracy_pct": acc_pct,
        "overall_kv_savings_pct": float(np.mean(all_savings)),
        "avg_prefill_ms": float(np.mean(all_prefill)),
        "avg_decode_tok_s": float(np.mean(all_decode)),
        "by_depth": summary_depth,
        "by_context_length": summary_ctx,
        "items": items_results,
    }


def generate_plots(all_results: Dict[str, Any], output_dir: str = "results") -> None:
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    configs = list(all_results.keys())

    # Plot 1: Overall Accuracy & Savings Bar Chart
    fig, ax1 = plt.subplots(figsize=(11, 6))
    x = np.arange(len(configs))
    width = 0.35

    accs = [all_results[c]["overall_accuracy_pct"] for c in configs]
    savings = [all_results[c]["overall_kv_savings_pct"] for c in configs]

    rects1 = ax1.bar(x - width / 2, accs, width, label="Recall Accuracy (%)", color="#1f77b4", alpha=0.9)
    ax1.set_ylabel("Recall Accuracy (%)", color="#1f77b4", fontsize=12, fontweight="bold")
    ax1.set_ylim(0, 115)
    ax1.set_xticks(x)
    ax1.set_xticklabels([c.replace(" ", "\n") for c in configs], fontsize=9)

    for r in rects1:
        h = r.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax2 = ax1.twinx()
    rects2 = ax2.bar(x + width / 2, savings, width, label="KV Cache Savings (%)", color="#2ca02c", alpha=0.8)
    ax2.set_ylabel("KV Cache Savings (%)", color="#2ca02c", fontsize=12, fontweight="bold")
    ax2.set_ylim(0, 115)

    for r in rects2:
        h = r.get_height()
        ax2.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.title("Phase 8 Ablation Study: Retrieval Accuracy & KV Memory Savings", fontsize=14, fontweight="bold", pad=15)
    fig.tight_layout()
    plot1_path = os.path.join(output_dir, "plot_phase8_ablation_accuracy.png")
    plt.savefig(plot1_path, dpi=300)
    plt.close()
    print(f"[Phase 8 Plotter] Saved: {plot1_path}")

    # Plot 2: Depth-Wise Recovery Curves
    depth_keys = ["start", "early_middle", "middle", "late_middle", "end"]
    depth_labels = ["Start (5%)", "Early-Mid (25%)", "Middle (50%)", "Late-Mid (75%)", "End (95%)"]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {
        "Full Baseline": "#333333",
        "StreamingLLM (S=4, W=256)": "#ff7f0e",
        "Pure Sliding Window (W=260)": "#9467bd",
        "H2O Heavy-Hitter (H=64, W=196)": "#d62728",
        "OGC-Cache (K=1, C=64)": "#2ca02c",
        "OGC-Cache (K=2, C=32)": "#17becf",
    }
    markers = ["o", "s", "^", "D", "v", "P"]

    for i, c in enumerate(configs):
        acc_vals = [all_results[c]["by_depth"][d]["accuracy_pct"] for d in depth_keys]
        color = colors.get(c, "#7f7f7f")
        lw = 2.5 if "OGC" in c or "Full" in c else 1.8
        ls = "-" if "OGC" in c or "Full" in c else "--"
        ax.plot(depth_labels, acc_vals, marker=markers[i % len(markers)], label=c,
                color=color, linewidth=lw, linestyle=ls, markersize=8)

    ax.set_title("Phase 8 Depth-Wise Fact Recovery Across Cache Paradigms", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Document Depth Location", fontsize=12, fontweight="bold")
    ax.set_ylabel("Retrieval Accuracy (%)", fontsize=12, fontweight="bold")
    ax.set_ylim(-5, 105)
    ax.legend(loc="upper left", frameon=True, fontsize=9)
    fig.tight_layout()
    plot2_path = os.path.join(output_dir, "plot_phase8_depth_recovery.png")
    plt.savefig(plot2_path, dpi=300)
    plt.close()
    print(f"[Phase 8 Plotter] Saved: {plot2_path}")

    # Plot 3: Pareto Frontier (KV Cache Size vs. Accuracy)
    fig, ax = plt.subplots(figsize=(10, 6))
    for c in configs:
        acc = all_results[c]["overall_accuracy_pct"]
        # KV cache size at L=4096
        if c == "Full Baseline":
            kv_size = 112.9
        else:
            kv_size = 7.11
        prefill = all_results[c]["avg_prefill_ms"]
        color = colors.get(c, "#1f77b4")
        ax.scatter([kv_size], [acc], s=160, color=color, zorder=5, edgecolors="black", linewidth=1.5, label=c)
        offset_y = 4 if "OGC" in c or "Full" in c else -6
        ax.annotate(f"{c}\n({acc:.1f}%, {kv_size:.1f} MiB)", xy=(kv_size, acc),
                    xytext=(0, offset_y), textcoords="offset points",
                    ha="center", va="bottom" if offset_y > 0 else "top", fontsize=8, fontweight="bold")

    ax.set_title("Pareto Trade-Off: KV Cache Memory at L=4096 vs. Retrieval Accuracy", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Sustained KV Cache Memory (MiB)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Overall Accuracy (%)", fontsize=12, fontweight="bold")
    ax.set_ylim(-5, 115)
    ax.set_xlim(0, 125)
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()
    plot3_path = os.path.join(output_dir, "plot_phase8_pareto_frontier.png")
    plt.savefig(plot3_path, dpi=300)
    plt.close()
    print(f"[Phase 8 Plotter] Saved: {plot3_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Phase 8 Ablation Study for OGC-Cache")
    parser.add_argument("--dataset", type=str, default="data/benchmark_dataset.json", help="Path to evaluation dataset")
    parser.add_argument("--plot-only", action="store_true", help="Re-generate comparison plots from existing results JSON")
    args = parser.parse_args()

    print("===========================================================================")
    print("PHASE 8 ABLATION STUDY: EVALUATING OBSERVATION-GUIDED CHUNKED CACHE (OGC)")
    print("===========================================================================")

    output_dir = "results"
    ablation_json_path = os.path.join(output_dir, "phase8_ablation_results.json")

    if args.plot_only:
        if not os.path.exists(ablation_json_path):
            print(f"Error: {ablation_json_path} does not exist to plot from.")
            sys.exit(1)
        print(f"[Phase 8] Regenerating plots from {ablation_json_path}...")
        with open(ablation_json_path, "r", encoding="utf-8") as f:
            all_results = json.load(f)
        generate_plots(all_results, output_dir=output_dir)
        print("[Phase 8] Plot generation complete.")
        return

    dataset = load_dataset(args.dataset)

    # Load baseline results from Phase 7 to avoid recomputing identical unmodified baselines
    p7_path = os.path.join(output_dir, "phase7_comparative_results.json")
    if not os.path.exists(p7_path):
        print(f"Error: {p7_path} not found. Run phase 7 first.")
        sys.exit(1)

    with open(p7_path, "r", encoding="utf-8") as f:
        all_results = json.load(f)

    # 1. Evaluate OGC-Cache (Config 1: K=1, C=64)
    print("\n[Phase 8] Initializing OGC-Cache Config 1 (K=1, C=64, W=192, S=4)...")
    engine_c1 = OGCCacheEngine(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        sink_tokens=4,
        recent_window=192,
        chunk_size=64,
        num_chunks=1,
        query_window=48,
        dtype=torch.bfloat16,
    )
    engine_c1.load_model()
    res_c1 = evaluate_ogc_config(engine_c1, dataset, "OGC-Cache (K=1, C=64)")
    all_results["OGC-Cache (K=1, C=64)"] = res_c1

    del engine_c1
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    # 2. Evaluate OGC-Cache (Config 2: K=2, C=32)
    print("\n[Phase 8] Initializing OGC-Cache Config 2 (K=2, C=32, W=192, S=4)...")
    engine_c2 = OGCCacheEngine(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        sink_tokens=4,
        recent_window=192,
        chunk_size=32,
        num_chunks=2,
        query_window=48,
        dtype=torch.bfloat16,
    )
    engine_c2.load_model()
    res_c2 = evaluate_ogc_config(engine_c2, dataset, "OGC-Cache (K=2, C=32)")
    all_results["OGC-Cache (K=2, C=32)"] = res_c2

    del engine_c2
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    # Save complete ablation results
    out_json = "results/phase8_ablation_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[Phase 8] Saved full ablation results to: {out_json}")

    # Generate comparative plots
    generate_plots(all_results)

    # Print summary table
    print("\n" + "=" * 95)
    print("PHASE 8 COMPREHENSIVE ABLATION STUDY RESULTS TABLE")
    print("=" * 95)
    print(f"{'Configuration':32s} {'Overall Acc (%)':17s} {'Accuracy Delta':16s} {'KV Savings (%)':16s} {'Avg Prefill (ms)'}")
    print("-" * 95)
    full_acc = all_results["Full Baseline"]["overall_accuracy_pct"]
    for name, res in all_results.items():
        acc = res["overall_accuracy_pct"]
        diff = acc - full_acc
        diff_str = f"{diff:+.2f} pp" if name != "Full Baseline" else "Reference"
        sav = res["overall_kv_savings_pct"]
        pref = res["avg_prefill_ms"]
        print(f"{name:32s} {acc:<17.2f} {diff_str:<16s} {sav:<16.2f} {pref:<15.1f}")
    print("=" * 95 + "\n")


if __name__ == "__main__":
    main()
