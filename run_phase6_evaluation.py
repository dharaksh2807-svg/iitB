"""
Phase 6 Full Empirical Evaluation: Memory Savings vs. Recall Degradation.
Executes the full experiment matrix across Uncompressed Baseline and 3 StreamingLLM Budgets.
Produces structured benchmark data, comparison tables, and publication-quality plots.
"""

import os
import time
import json
import argparse
from typing import Dict, Any, List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.baseline import UncompressedBaselineEngine
from src.streaming_cache import StreamingLLMEngine
from src.dataset import score_response


def evaluate_engine_on_dataset(engine: Any, items: List[Dict[str, Any]], config_name: str) -> Dict[str, Any]:
    print(f"\n{'='*70}\n[Phase6] Evaluating {config_name} across {len(items)} benchmark items...\n{'='*70}")
    
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

        print(f"  [{idx+1}/{total_count}] {config_name} | {item_id} (L={item['actual_token_length']}, Depth={depth})...", end=" ", flush=True)

        t0 = time.time()
        res = engine.generate(prompt=doc, max_new_tokens=35, do_sample=False)
        pred = res.output_text

        is_correct, score_frac, _ = score_response(pred, gts)
        if is_correct:
            correct_count += 1

        status_str = "PASS" if is_correct else "FAIL"
        uncomp_b = res.uncompressed_kv_bytes if hasattr(res, "uncompressed_kv_bytes") else getattr(res, "theoretical_kv_bytes", 0)
        comp_b = res.compressed_kv_bytes if hasattr(res, "compressed_kv_bytes") else getattr(res, "kv_cache_bytes", 0)
        savings = getattr(res, "kv_cache_savings_pct", 0.0)

        total_prefill_ms += res.prefill_time_ms
        total_decode_ms += res.decode_time_ms
        total_tokens_gen += res.decode_steps
        total_uncompressed_bytes += uncomp_b
        total_compressed_bytes += comp_b

        print(f"[{status_str}] KV: {comp_b / (1024**2):.2f} MiB ({savings:.1f}% sav) | Prefill: {res.prefill_time_ms:.1f}ms | Dec: {res.decode_tokens_per_sec:.1f} tok/s")

        # Category tracking
        if depth not in depth_stats:
            depth_stats[depth] = {"correct": 0, "total": 0}
        depth_stats[depth]["total"] += 1
        if is_correct:
            depth_stats[depth]["correct"] += 1

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

        evaluated.append({
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
        })

    overall_accuracy = (correct_count / total_count * 100.0) if total_count > 0 else 0.0
    overall_savings = (100.0 * (total_uncompressed_bytes - total_compressed_bytes) / total_uncompressed_bytes) if total_uncompressed_bytes > 0 else 0.0
    avg_prefill_ms = total_prefill_ms / total_count if total_count > 0 else 0.0
    avg_decode_tok_s = (total_tokens_gen / (total_decode_ms / 1000.0)) if total_decode_ms > 0 else 0.0

    # Depth breakdown
    depth_summary = {
        d: {
            "correct": stats["correct"],
            "total": stats["total"],
            "accuracy_pct": round(stats["correct"] / stats["total"] * 100.0, 2)
        }
        for d, stats in depth_stats.items()
    }

    # Context breakdown
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


def generate_publication_plots(results: Dict[str, Any], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    
    contexts = [1024, 2048, 4096]
    configs = list(results.keys())
    colors = {
        "Full Baseline": "#1f77b4",
        "StreamingLLM (B=260)": "#ff7f0e",
        "StreamingLLM (B=516)": "#2ca02c",
        "StreamingLLM (B=1028)": "#d62728",
    }

    # ----------------------------------------------------
    # Plot 1: KV-Cache Memory Savings (%) vs Context Length
    # ----------------------------------------------------
    plt.figure(figsize=(8, 5), dpi=300)
    for cfg in configs:
        if "Baseline" in cfg:
            continue
        sav_vals = [results[cfg]["by_context_length"][str(c)]["kv_savings_pct"] for c in contexts]
        plt.plot(contexts, sav_vals, marker="o", linewidth=2.5, markersize=8, label=cfg, color=colors.get(cfg))

    plt.title("KV-Cache Memory Savings (%) vs. Context Length", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Input Context Length (Tokens)", fontsize=12)
    plt.ylabel("Physical KV-Cache Savings (%)", fontsize=12)
    plt.ylim(0, 100)
    plt.xticks(contexts, [f"{c:,}" for c in contexts])
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(frameon=True, facecolor="white", loc="lower right")
    plt.tight_layout()
    p1 = os.path.join(output_dir, "plot_memory_savings_vs_context.png")
    plt.savefig(p1)
    plt.close()
    print(f"[Plotter] Saved: {p1}")

    # ----------------------------------------------------
    # Plot 2: Recall Accuracy (%) vs. Cache Budget
    # ----------------------------------------------------
    plt.figure(figsize=(8, 5), dpi=300)
    budgets = [260, 516, 1028, 4096]
    budget_labels = ["B=260", "B=516", "B=1028", "Full Cache"]
    acc_by_budget = [
        results["StreamingLLM (B=260)"]["overall_accuracy_pct"],
        results["StreamingLLM (B=516)"]["overall_accuracy_pct"],
        results["StreamingLLM (B=1028)"]["overall_accuracy_pct"],
        results["Full Baseline"]["overall_accuracy_pct"],
    ]

    bars = plt.bar(budget_labels, acc_by_budget, color=["#ff7f0e", "#2ca02c", "#d62728", "#1f77b4"], width=0.55, edgecolor="black", linewidth=1.2)
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, h + 1.5, f"{h:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

    plt.title("Long-Context Recall Accuracy (%) vs. Cache Budget", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("KV-Cache Budget Configuration", fontsize=12)
    plt.ylabel("Overall Recall Accuracy (%)", fontsize=12)
    plt.ylim(0, 115)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    p2 = os.path.join(output_dir, "plot_recall_accuracy_vs_budget.png")
    plt.savefig(p2)
    plt.close()
    print(f"[Plotter] Saved: {p2}")

    # ----------------------------------------------------
    # Plot 3: Accuracy Change (pp) vs. Memory Reduction (%) - Pareto Frontier
    # ----------------------------------------------------
    plt.figure(figsize=(8, 5), dpi=300)
    base_acc = results["Full Baseline"]["overall_accuracy_pct"]

    for cfg in configs:
        sav = results[cfg]["overall_kv_savings_pct"]
        acc = results[cfg]["overall_accuracy_pct"]
        deg = acc - base_acc  # pp change
        plt.scatter(sav, deg, s=180, color=colors.get(cfg, "#333333"), label=cfg, edgecolors="black", linewidth=1.5, zorder=5)
        plt.annotate(
            f"{cfg}\n(Sav: {sav:.1f}%, $\\Delta$: {deg:+.1f}pp)",
            (sav, deg),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
            fontweight="bold"
        )

    plt.axhline(0, color="gray", linestyle="--", linewidth=1.2)
    plt.title("Accuracy Change (percentage points) vs. KV Memory Savings (%)", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("KV-Cache Memory Savings (%)", fontsize=12)
    plt.ylabel("Accuracy Change relative to Full Baseline (pp)", fontsize=12)
    plt.xlim(-5, 100)
    plt.ylim(-60, 15)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    p3 = os.path.join(output_dir, "plot_accuracy_change_vs_memory_reduction.png")
    plt.savefig(p3)
    plt.close()
    print(f"[Plotter] Saved: {p3}")

    # ----------------------------------------------------
    # Plot 4: Recall Accuracy by Document Depth (Start -> End)
    # ----------------------------------------------------
    plt.figure(figsize=(9, 5), dpi=300)
    depths = ["start", "early_middle", "middle", "late_middle", "end"]
    depth_labels = ["Start (5%)", "Early-Mid (25%)", "Middle (50%)", "Late-Mid (75%)", "End (95%)"]
    x = np.arange(len(depths))
    width = 0.2

    for i, cfg in enumerate(configs):
        y_vals = [results[cfg]["by_depth"].get(d, {}).get("accuracy_pct", 0.0) for d in depths]
        plt.bar(x + (i - 1.5) * width, y_vals, width, label=cfg, color=colors.get(cfg), edgecolor="black", linewidth=0.8)

    plt.title("Recall Accuracy Across Document Depths: Baseline vs. Compressed", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Document Insertion Depth", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.xticks(x, depth_labels, fontsize=10)
    plt.ylim(0, 120)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.legend(frameon=True, facecolor="white", loc="upper left")
    plt.tight_layout()
    p4 = os.path.join(output_dir, "plot_depth_recall_breakdown.png")
    plt.savefig(p4)
    plt.close()
    print(f"[Plotter] Saved: {p4}")


def main():
    from run_research_benchmark import main as run_fresh
    run_fresh(default_policies=("full", "streaming"))


if __name__ == "__main__":
    main()
