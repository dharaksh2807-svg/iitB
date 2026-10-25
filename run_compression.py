"""
CLI Runner for StreamingLLM Cache Compression.
Executes compressed inference and outputs memory savings comparison.
"""

import os
import json
import argparse
from src.streaming_cache import StreamingLLMEngine


def main():
    parser = argparse.ArgumentParser(description="Run StreamingLLM KV-cache compression.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="Model Hub ID")
    parser.add_argument("--device", type=str, default="mps", help="Device (mps, cpu)")
    parser.add_argument("--sinks", type=int, default=4, help="Attention sink tokens")
    parser.add_argument("--window", type=int, default=256, help="Recent rolling window tokens")
    parser.add_argument("--max_tokens", type=int, default=30, help="Max new tokens to generate")
    args = parser.parse_args()

    print("=" * 70)
    print("PHASE 5: STREAMINGLLM CACHE COMPRESSION EXECUTION")
    print("=" * 70)
    print(f"Configuration: Sinks (S) = {args.sinks} | Window (W) = {args.window} | Total Budget = {args.sinks + args.window}")

    engine = StreamingLLMEngine(
        model_name_or_path=args.model,
        device=args.device,
        sink_tokens=args.sinks,
        window_tokens=args.window,
    )
    meta = engine.load_model()

    # Load an evaluation prompt from pilot dataset
    with open("data/pilot_dataset.json") as f:
        pilot_items = json.load(f)

    # Item 4 has needle near end (95% depth)
    item = pilot_items[4]
    print(f"\nEvaluating Item: {item['item_id']} (Target: {item['actual_token_length']} tokens, Depth: {item['depth_category']})")
    print(f"Ground Truth: {item['ground_truth_answers']}")

    res = engine.generate(prompt=item["document"], max_new_tokens=args.max_tokens, do_sample=False)

    print("\n[Compressed Inference Result]")
    print(f"Generated text:\n{res.output_text}\n")
    print(f"Input tokens: {res.input_tokens}")
    print(f"Generated tokens: {res.generated_tokens}")
    print(f"Actual cached tokens: {res.actual_cached_tokens} (Budget: {res.cache_budget_tokens})")
    print(f"Prefill latency: {res.prefill_time_ms:.2f} ms")
    print(f"Decoding throughput: {res.decode_tokens_per_sec:.2f} tokens/sec")
    print(f"Uncompressed KV cache: {res.uncompressed_kv_bytes / (1024**2):.2f} MiB")
    print(f"Compressed KV cache:   {res.compressed_kv_mib:.2f} MiB ({res.compressed_kv_bytes:,} bytes)")
    print(f"KV CACHE MEMORY SAVINGS: {res.kv_cache_savings_pct:.2f}%")
    print(f"Peak Device Allocated: {res.peak_device_allocated_mib:.2f} MiB")

    # Checkpoint analysis
    cps = res.checkpoints
    print("\n[Memory Lifecycle Checkpoints]")
    print(f"  1. Start:         KV Cache: {cps['inference_start']['kv_cache_mib']:.2f} MiB | Device: {cps['inference_start']['device_allocated_mib']:.2f} MiB")
    print(f"  2. Peak Prefill:  KV Cache: {cps['peak_prefill_raw']['kv_cache_mib']:.2f} MiB | Device: {cps['peak_prefill_raw']['device_allocated_mib']:.2f} MiB")
    print(f"  3. Pruned Cache:  KV Cache: {cps['post_prefill_pruned']['kv_cache_mib']:.2f} MiB | Device: {cps['post_prefill_pruned']['device_allocated_mib']:.2f} MiB")
    print(f"  4. Post Decoding: KV Cache: {cps['post_decoding']['kv_cache_mib']:.2f} MiB | Device: {cps['post_decoding']['device_allocated_mib']:.2f} MiB")

    os.makedirs("results", exist_ok=True)
    out_file = "results/streamingllm_verification.json"
    with open(out_file, "w") as f:
        json.dump(res.to_dict(), f, indent=2)
    print(f"\nSaved verification results to {out_file}")


if __name__ == "__main__":
    main()
