"""
Benchmark Evaluator Harness.
Evaluates an inference engine (baseline or compressed) against the long-context recall benchmark,
computing overall recall accuracy, depth-segmented accuracy, and latency metrics.
"""

import os
import json
import argparse
from typing import Dict, Any, List, Optional
from src.baseline import UncompressedBaselineEngine
from src.dataset import score_response


def run_evaluation(
    engine: Any,
    dataset_path: str,
    max_new_tokens: int = 35,
    max_items: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Evaluates engine across items in dataset_path.
    Returns structured results dictionary.
    """
    with open(dataset_path, "r") as f:
        items = json.load(f)

    if max_items is not None and max_items > 0:
        items = items[:max_items]

    print(f"\n[Evaluator] Running benchmark evaluation on {len(items)} items from {dataset_path}...")
    evaluated_items: List[Dict[str, Any]] = []

    correct_count = 0
    total_count = len(items)

    depth_stats: Dict[str, Dict[str, int]] = {}
    query_type_stats: Dict[str, Dict[str, int]] = {}
    context_stats: Dict[str, Dict[str, int]] = {}

    for idx, item in enumerate(items):
        item_id = item["item_id"]
        ctx_len = item["target_context_length"]
        depth = item["depth_category"]
        q_type = item["query_type"]
        gts = item["ground_truth_answers"]
        doc = item["document"]

        print(f"  [{idx+1}/{total_count}] Testing {item_id} (Length: {item['actual_token_length']} | Depth: {depth} | Type: {q_type})...")
        
        # Run inference
        inf_res = engine.generate(prompt=doc, max_new_tokens=max_new_tokens, do_sample=False)
        pred = inf_res.output_text

        # Score
        is_correct, score_frac, score_meta = score_response(pred, gts)
        if is_correct:
            correct_count += 1

        # Track per-category stats
        for cat_dict, key in [(depth_stats, depth), (query_type_stats, q_type), (context_stats, str(ctx_len))]:
            if key not in cat_dict:
                cat_dict[key] = {"correct": 0, "total": 0}
            cat_dict[key]["total"] += 1
            if is_correct:
                cat_dict[key]["correct"] += 1

        status_str = "CORRECT" if is_correct else "INCORRECT"
        print(f"      -> Answer: '{pred.strip()[:60]}...' | Status: {status_str} | KV Cache: {inf_res.kv_cache_mib:.2f} MiB")

        evaluated_items.append({
            "item_id": item_id,
            "context_length": ctx_len,
            "depth_category": depth,
            "depth_fraction": item["depth_fraction"],
            "query_type": q_type,
            "ground_truths": gts,
            "prediction": pred,
            "is_correct": is_correct,
            "score_fraction": score_frac,
            "prefill_time_ms": inf_res.prefill_time_ms,
            "decode_time_ms": inf_res.decode_time_ms,
            "kv_cache_mib": inf_res.kv_cache_mib,
        })

    overall_accuracy = (correct_count / total_count * 100.0) if total_count > 0 else 0.0

    # Calculate category percentages
    def calc_pcts(d: Dict[str, Dict[str, int]]) -> Dict[str, Any]:
        return {
            k: {
                "correct": v["correct"],
                "total": v["total"],
                "accuracy_pct": round((v["correct"] / v["total"] * 100.0) if v["total"] > 0 else 0.0, 2)
            }
            for k, v in d.items()
        }

    summary = {
        "dataset_path": dataset_path,
        "total_evaluated": total_count,
        "total_correct": correct_count,
        "overall_accuracy_pct": round(overall_accuracy, 2),
        "by_depth": calc_pcts(depth_stats),
        "by_query_type": calc_pcts(query_type_stats),
        "by_context_length": calc_pcts(context_stats),
        "items": evaluated_items,
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate benchmark dataset on baseline or compressed engine.")
    parser.add_argument("--dataset", type=str, default="data/pilot_dataset.json", help="Dataset path")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="Model Hub ID")
    parser.add_argument("--device", type=str, default="mps", help="Device (mps, cpu)")
    parser.add_argument("--max_tokens", type=int, default=35, help="Max output tokens")
    parser.add_argument("--max_items", type=int, default=None, help="Max items to evaluate")
    parser.add_argument("--output", type=str, default=None, help="Output JSON results path")
    args = parser.parse_args()

    print("=" * 70)
    print("PHASE 4: BENCHMARK RECALL EVALUATION")
    print("=" * 70)

    # Initialize baseline engine
    engine = UncompressedBaselineEngine(model_name_or_path=args.model, device=args.device)
    engine.load_model()

    summary = run_evaluation(
        engine=engine,
        dataset_path=args.dataset,
        max_new_tokens=args.max_tokens,
        max_items=args.max_items,
    )

    print("\n" + "=" * 70)
    print(f"BENCHMARK RESULTS: {args.dataset}")
    print("=" * 70)
    print(f"Overall Recall Accuracy: {summary['overall_accuracy_pct']}% ({summary['total_correct']}/{summary['total_evaluated']})")
    
    print("\nAccuracy by Document Depth:")
    for depth, dstats in summary["by_depth"].items():
        print(f"  - {depth:<15}: {dstats['accuracy_pct']:>6.1f}% ({dstats['correct']}/{dstats['total']})")

    print("\nAccuracy by Query Type:")
    for qtype, qstats in summary["by_query_type"].items():
        print(f"  - {qtype:<15}: {qstats['accuracy_pct']:>6.1f}% ({qstats['correct']}/{qstats['total']})")

    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved evaluation results to {args.output}")


if __name__ == "__main__":
    main()
