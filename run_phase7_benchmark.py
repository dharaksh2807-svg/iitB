"""
Phase 7 Multi-Baseline Benchmark:
Controlled Empirical Comparison of Four Distinct KV-Cache Strategies:
1. Full Uncompressed Baseline (B = L)
2. Pure Sliding-Window Cache (S = 0, W = 260)
3. StreamingLLM Sink Window Cache (S = 4, W = 256)
4. H2O Heavy-Hitter Oracle Cache (H = 64, W = 196)

All three compression policies operate at the exact identical cache budget (B = 260 tokens, 7.11 MiB).
Identical model (Qwen2.5-1.5B-Instruct), identical 30-item benchmark, identical greedy decoding.
"""

import os
import gc
import time
import json
import argparse
from typing import Dict, Any, List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.baseline import UncompressedBaselineEngine
from src.streaming_cache import StreamingLLMEngine
from src.sliding_cache import PureSlidingWindowEngine
from src.h2o_cache import H2OCacheEngine
from src.dataset import score_response


def evaluate_engine_on_dataset(
    engine: Any, items: List[Dict[str, Any]], config_name: str
) -> Dict[str, Any]:
    print(f"\n{'='*75}\n[Phase7] Evaluating {config_name} across {len(items)} items...\n{'='*75}")
    evaluated: List[Dict[str, Any]] = []
    correct_count = 0
    total_count = len(items)

    depth_stats: Dict[str, Dict[str, int]] = {}
    context_stats: Dict[str, Dict[str, Any]] = {}

    total_prefill_ms = 0.0
    total_decode_ms = 0.0
    total_tokens_gen = 0
    total_uncompressed_bytes = 0
    total_compressed_bytes = 0

    for idx, item in enumerate(items):
        item_id = item["item_id"]
        ctx_len = item["target_context_length"]
        depth = item["depth_category"]
        depth_frac = item["depth_fraction"]
        q_type = item["query_type"]
        gts = item["ground_truth_answers"]
        doc = item["document"]

        print(
            f"  [{idx+1:2d}/{total_count}] {config_name} | {item_id} (L={item['actual_token_length']}, Depth={depth})...",
            end=" ",
            flush=True,
        )

        res = engine.generate(prompt=doc, max_new_tokens=35, do_sample=False)
        pred = res.output_text

        is_correct, _, _ = score_response(pred, gts)
        if is_correct:
            correct_count += 1

        status_str = "PASS" if is_correct else "FAIL"
        uncomp_b = (
            res.uncompressed_kv_bytes
            if hasattr(res, "uncompressed_kv_bytes")
            else getattr(res, "theoretical_kv_bytes", 0)
        )
        comp_b = (
            res.compressed_kv_bytes
            if hasattr(res, "compressed_kv_bytes")
            else getattr(res, "kv_cache_bytes", 0)
        )
        savings = getattr(res, "kv_cache_savings_pct", 0.0)

        total_prefill_ms += res.prefill_time_ms
        total_decode_ms += res.decode_time_ms
        total_tokens_gen += res.generated_tokens
        total_uncompressed_bytes += uncomp_b
        total_compressed_bytes += comp_b

        print(
            f"[{status_str}] KV: {comp_b / (1024**2):.2f} MiB ({savings:.1f}% sav) | "
            f"Prefill: {res.prefill_time_ms:.1f}ms | Dec: {res.decode_tokens_per_sec:.1f} tok/s"
        )

        # Depth tracking
        if depth not in depth_stats:
            depth_stats[depth] = {"correct": 0, "total": 0}
        depth_stats[depth]["total"] += 1
        if is_correct:
            depth_stats[depth]["correct"] += 1

        # Context tracking
        ctx_key = str(ctx_len)
        if ctx_key not in context_stats:
            context_stats[ctx_key] = {
                "correct": 0,
                "total": 0,
                "uncompressed_bytes": 0,
                "compressed_bytes": 0,
                "prefill_ms": 0.0,
                "decode_tokens_per_sec": 0.0,
            }
        context_stats[ctx_key]["total"] += 1
        if is_correct:
            context_stats[ctx_key]["correct"] += 1
        context_stats[ctx_key]["uncompressed_bytes"] += uncomp_b
        context_stats[ctx_key]["compressed_bytes"] += comp_b
        context_stats[ctx_key]["prefill_ms"] += res.prefill_time_ms
        context_stats[ctx_key]["decode_tokens_per_sec"] += res.decode_tokens_per_sec

        item_record = {
            "item_id": item_id,
            "context_length": ctx_len,
            "depth_category": depth,
            "depth_fraction": depth_frac,
            "query_type": q_type,
            "ground_truths": gts,
            "prediction": pred.strip(),
            "is_correct": is_correct,
            "prefill_time_ms": res.prefill_time_ms,
            "decode_tokens_per_sec": res.decode_tokens_per_sec,
            "uncompressed_kv_bytes": uncomp_b,
            "compressed_kv_bytes": comp_b,
            "kv_cache_savings_pct": savings,
            "peak_device_allocated_mib": res.peak_device_allocated_mib,
        }
        if hasattr(res, "retained_heavy_hitter_indices"):
            item_record["heavy_hitters_count"] = len(res.retained_heavy_hitter_indices)
        evaluated.append(item_record)

    overall_accuracy = (correct_count / total_count * 100.0) if total_count > 0 else 0.0
    overall_savings = (
        (100.0 * (total_uncompressed_bytes - total_compressed_bytes) / total_uncompressed_bytes)
        if total_uncompressed_bytes > 0
        else 0.0
    )
    avg_prefill_ms = total_prefill_ms / total_count if total_count > 0 else 0.0
    avg_decode_tok_s = (total_tokens_gen / (total_decode_ms / 1000.0)) if total_decode_ms > 0 else 0.0

    depth_summary = {
        d: {
            "correct": stats["correct"],
            "total": stats["total"],
            "accuracy_pct": round(stats["correct"] / stats["total"] * 100.0, 2),
        }
        for d, stats in depth_stats.items()
    }

    context_summary = {}
    for c, stats in context_stats.items():
        n = stats["total"]
        u_b = stats["uncompressed_bytes"]
        c_b = stats["compressed_bytes"]
        sav = (100.0 * (u_b - c_b) / u_b) if u_b > 0 else 0.0
        context_summary[c] = {
            "correct": stats["correct"],
            "total": n,
            "accuracy_pct": round(stats["correct"] / n * 100.0, 2),
            "uncompressed_mib": round(u_b / n / (1024**2), 2),
            "compressed_mib": round(c_b / n / (1024**2), 2),
            "kv_savings_pct": round(sav, 2),
            "avg_prefill_ms": round(stats["prefill_ms"] / n, 2),
            "avg_decode_tok_s": round(stats["decode_tokens_per_sec"] / n, 2),
        }

    return {
        "config_name": config_name,
        "total_evaluated": total_count,
        "total_correct": correct_count,
        "overall_accuracy_pct": round(overall_accuracy, 2),
        "overall_kv_savings_pct": round(overall_savings, 2),
        "avg_prefill_ms": round(avg_prefill_ms, 2),
        "avg_decode_tok_s": round(avg_decode_tok_s, 2),
        "by_depth": depth_summary,
        "by_context_length": context_summary,
        "items": evaluated,
    }


def generate_phase7_comparative_plots(results: Dict[str, Any], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    configs = list(results.keys())
    colors = {
        "Full Baseline": "#1f77b4",
        "Pure Sliding Window (W=260)": "#9467bd",
        "StreamingLLM (S=4, W=256)": "#ff7f0e",
        "H2O Heavy-Hitter (H=64, W=196)": "#2ca02c",
    }

    # ----------------------------------------------------
    # Plot 1: Policy Comparison at Identical Budget B=260
    # ----------------------------------------------------
    plt.figure(figsize=(9, 5.5), dpi=300)
    accs = [results[cfg]["overall_accuracy_pct"] for cfg in configs]
    bars = plt.bar(
        configs,
        accs,
        color=[colors.get(c, "#555555") for c in configs],
        width=0.55,
        edgecolor="black",
        linewidth=1.2,
    )
    for bar in bars:
        h = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 1.5,
            f"{h:.1f}%",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    plt.title(
        "Overall Long-Context Recall Accuracy: Full Baseline vs. 3 Eviction Policies (B=260)",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )
    plt.ylabel("Recall Accuracy (%)", fontsize=12)
    plt.ylim(0, 115)
    plt.xticks(rotation=12, ha="right", fontsize=10, fontweight="bold")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    p1 = os.path.join(output_dir, "plot_phase7_policy_accuracy_comparison.png")
    plt.savefig(p1)
    plt.close()
    print(f"[Phase7 Plotter] Saved: {p1}")

    # ----------------------------------------------------
    # Plot 2: Depth-Wise Recall by Cache Eviction Policy
    # ----------------------------------------------------
    plt.figure(figsize=(10, 5.5), dpi=300)
    depths = ["start", "early_middle", "middle", "late_middle", "end"]
    depth_labels = ["Start (5%)", "Early-Mid (25%)", "Middle (50%)", "Late-Mid (75%)", "End (95%)"]
    x = np.arange(len(depths))
    width = 0.2

    for i, cfg in enumerate(configs):
        y_vals = [results[cfg]["by_depth"].get(d, {}).get("accuracy_pct", 0.0) for d in depths]
        plt.bar(
            x + (i - 1.5) * width,
            y_vals,
            width,
            label=cfg,
            color=colors.get(cfg, "#555555"),
            edgecolor="black",
            linewidth=0.8,
        )

    plt.title(
        "Depth-Wise Recall Breakdown: Comparing KV-Cache Eviction Mechanisms",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )
    plt.xlabel("Needle Insertion Depth in Document", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.xticks(x, depth_labels, fontsize=10, fontweight="bold")
    plt.ylim(0, 120)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.legend(frameon=True, facecolor="white", loc="upper left", fontsize=9)
    plt.tight_layout()
    p2 = os.path.join(output_dir, "plot_phase7_depth_breakdown_by_policy.png")
    plt.savefig(p2)
    plt.close()
    print(f"[Phase7 Plotter] Saved: {p2}")

    # ----------------------------------------------------
    # Plot 3: Prefill Latency and Decoding Throughput
    # ----------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(9, 5), dpi=300)
    ax2 = ax1.twinx()

    x_indices = np.arange(len(configs))
    prefill_vals = [results[cfg]["avg_prefill_ms"] for cfg in configs]
    decode_vals = [results[cfg]["avg_decode_tok_s"] for cfg in configs]

    w = 0.35
    b1 = ax1.bar(x_indices - w / 2, prefill_vals, width=w, color="#3498db", label="Prefill Latency (ms)", edgecolor="black")
    b2 = ax2.bar(x_indices + w / 2, decode_vals, width=w, color="#e67e22", label="Decode Throughput (tok/s)", edgecolor="black")

    ax1.set_ylabel("Average Prefill Latency (ms)", color="#2980b9", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Decode Throughput (tok/sec)", color="#d35400", fontsize=11, fontweight="bold")
    ax1.set_xticks(x_indices)
    ax1.set_xticklabels(configs, rotation=12, ha="right", fontsize=9, fontweight="bold")
    ax1.set_title("Inference Efficiency Across Cache Strategies", fontsize=13, fontweight="bold", pad=12)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", frameon=True, facecolor="white")

    plt.tight_layout()
    p3 = os.path.join(output_dir, "plot_phase7_latency_throughput_comparison.png")
    plt.savefig(p3)
    plt.close()
    print(f"[Phase7 Plotter] Saved: {p3}")


def main():
    parser = argparse.ArgumentParser(description="Execute Phase 7 Multi-Baseline Benchmark.")
    parser.add_argument("--dataset", type=str, default="data/benchmark_dataset.json", help="Dataset path")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="Model Hub ID")
    parser.add_argument("--device", type=str, default="mps", help="Device (mps, cpu)")
    args = parser.parse_args()

    with open(args.dataset, "r") as f:
        items = json.load(f)

    print(f"Loaded {len(items)} items from {args.dataset}")
    all_results: Dict[str, Any] = {}

    # Load Full Baseline and StreamingLLM results from Phase 6
    phase6_path = "results/phase6_benchmark_results.json"
    if os.path.exists(phase6_path):
        with open(phase6_path, "r") as f:
            p6_data = json.load(f)
            if "Full Baseline" in p6_data:
                all_results["Full Baseline"] = p6_data["Full Baseline"]
            if "StreamingLLM (B=260)" in p6_data:
                # Rename for clean comparison label
                all_results["StreamingLLM (S=4, W=256)"] = p6_data["StreamingLLM (B=260)"]
                all_results["StreamingLLM (S=4, W=256)"]["config_name"] = "StreamingLLM (S=4, W=256)"

    # ----------------------------------------------------
    # Baseline 1: Pure Sliding Window (S=0, W=260)
    # ----------------------------------------------------
    print("\n[Phase7] Initializing Baseline 1: Pure Sliding Window (W=260)...")
    engine_sw = PureSlidingWindowEngine(model_name_or_path=args.model, device=args.device, window_tokens=260)
    engine_sw.load_model()
    all_results["Pure Sliding Window (W=260)"] = evaluate_engine_on_dataset(
        engine_sw, items, "Pure Sliding Window (W=260)"
    )
    # Explicit garbage collection to maintain host memory hygiene
    del engine_sw
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    time.sleep(2)

    # ----------------------------------------------------
    # Baseline 2: H2O Heavy-Hitter Cache (H=64, W=196)
    # ----------------------------------------------------
    print("\n[Phase7] Initializing Baseline 2: H2O Heavy-Hitter Cache (H=64, W=196)...")
    engine_h2o = H2OCacheEngine(
        model_name_or_path=args.model,
        device=args.device,
        heavy_hitters=64,
        recent_tokens=196,
    )
    engine_h2o.load_model()
    all_results["H2O Heavy-Hitter (H=64, W=196)"] = evaluate_engine_on_dataset(
        engine_h2o, items, "H2O Heavy-Hitter (H=64, W=196)"
    )
    del engine_h2o
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    # Save full comparative results
    os.makedirs("results", exist_ok=True)
    out_json = "results/phase7_comparative_results.json"
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[Phase7] Saved full multi-baseline comparison results to: {out_json}")

    # Generate comparative plots
    generate_phase7_comparative_plots(all_results, output_dir="results")

    # Print Summary Table
    print("\n" + "=" * 90)
    print("PHASE 7 MULTI-BASELINE COMPARATIVE RESULTS TABLE")
    print("=" * 90)
    print(f"{'Configuration':<32}{'Overall Acc (%)':<16}{'Accuracy $\\Delta$ (pp)':<18}{'KV Savings (%)':<18}{'Avg Prefill (ms)':<16}")
    print("-" * 90)
    base_acc = all_results["Full Baseline"]["overall_accuracy_pct"]
    for cfg, data in all_results.items():
        deg = data["overall_accuracy_pct"] - base_acc
        print(
            f"{cfg:<32}"
            f"{data['overall_accuracy_pct']:<16.2f}"
            f"{deg:<+18.2f}"
            f"{data['overall_kv_savings_pct']:<18.2f}"
            f"{data['avg_prefill_ms']:<16.1f}"
        )
    print("=" * 90)


if __name__ == "__main__":
    main()
