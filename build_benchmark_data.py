"""
Build and persist the reproducible benchmark dataset files.
"""

import os
import json
from transformers import AutoTokenizer
from src.dataset import BenchmarkGenerator


def main():
    print("=" * 70)
    print("PHASE 4: BUILDING DETERMINISTIC LONG-CONTEXT RECALL DATASETS")
    print("=" * 70)

    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    print(f"[DatasetBuilder] Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

    generator = BenchmarkGenerator(tokenizer=tokenizer, base_seed=42)
    os.makedirs("data", exist_ok=True)

    # 1. Build Pilot Dataset (5 items, 1024 tokens, covering all 5 depths)
    print("\n[DatasetBuilder] Generating pilot evaluation suite (5 items at 1024 tokens)...")
    pilot_items = generator.generate_pilot_suite()
    pilot_path = "data/pilot_dataset.json"
    with open(pilot_path, "w") as f:
        json.dump([item.to_dict() for item in pilot_items], f, indent=2)
    print(f"  -> Saved {len(pilot_items)} pilot items to {pilot_path}")

    for idx, item in enumerate(pilot_items):
        print(f"     [{idx+1}] ID: {item.item_id} | Depth: {item.depth_category} ({item.depth_fraction*100:.0f}%) | Tokens: {item.actual_token_length} | GT: {item.ground_truth_answers}")

    # 2. Build Full Benchmark Dataset (Contexts: 1024, 2048, 4096 x 5 depths x 2 query types = 30 items)
    print("\n[DatasetBuilder] Generating full benchmark suite (Contexts: 1024, 2048, 4096 across 5 depths)...")
    benchmark_items = generator.generate_benchmark_suite(context_lengths=[1024, 2048, 4096])
    bench_path = "data/benchmark_dataset.json"
    with open(bench_path, "w") as f:
        json.dump([item.to_dict() for item in benchmark_items], f, indent=2)
    print(f"  -> Saved {len(benchmark_items)} full benchmark items to {bench_path}")

    # Summary table
    print("\nBenchmark Composition Summary:")
    print(f"{'Target Tokens':<16}{'Depth':<16}{'Query Type':<16}{'Actual Tokens':<16}{'Ground Truth':<20}")
    print("-" * 75)
    for item in benchmark_items[:10]:
        gt_str = ", ".join(item.ground_truth_answers)
        print(f"{item.target_context_length:<16}{item.depth_category:<16}{item.query_type:<16}{item.actual_token_length:<16}{gt_str:<20}")
    print(f"... ({len(benchmark_items)-10} additional items not shown)")


if __name__ == "__main__":
    main()
