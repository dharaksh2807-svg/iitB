"""
CLI Runner and Benchmark Script for Uncompressed Baseline.
Executes verification inference and context length scaling suite.
"""

import os
import json
import argparse
from src.baseline import UncompressedBaselineEngine


def main():
    parser = argparse.ArgumentParser(description="Run uncompressed KV-cache baseline.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="Model Hub ID")
    parser.add_argument("--device", type=str, default="mps", help="Device (mps, cpu, cuda)")
    parser.add_argument("--max_tokens", type=int, default=30, help="Max new tokens to generate")
    parser.add_argument("--scaling", action="store_true", help="Run context scaling benchmark")
    args = parser.parse_args()

    print("=" * 70)
    print("PHASE 3: UNCOMPRESSED BASELINE INFERENCE & INSTRUMENTATION")
    print("=" * 70)

    # 1. Initialize and load model
    engine = UncompressedBaselineEngine(model_name_or_path=args.model, device=args.device)
    meta = engine.load_model()
    print("\nModel Metadata:")
    print(json.dumps(meta, indent=2))

    # 2. Run reference prompt
    test_prompt = (
        "Project specification overview:\n"
        "Project Code: ALPHA-9042\n"
        "Assigned Database Port: 5432\n"
        "Deployment Target: Kubernetes US-East\n"
        "Status: Operational\n\n"
        "Question: What is the Project Code and Assigned Database Port? Answer concisely:"
    )

    print("\n--- Running Verification Inference ---")
    result = engine.generate(prompt=test_prompt, max_new_tokens=args.max_tokens, do_sample=False)
    print("\n[Inference Result Output]")
    print(f"Generated text:\n{result.output_text}\n")
    print(f"Input tokens: {result.input_tokens}")
    print(f"Generated tokens: {result.generated_tokens}")
    print(f"Prefill latency: {result.prefill_time_ms:.2f} ms")
    print(f"Decoding throughput: {result.decode_tokens_per_sec:.2f} tokens/sec")
    print(f"Actual KV cache memory: {result.kv_cache_mib:.4f} MiB ({result.kv_cache_bytes:,} bytes)")
    print(f"Theoretical KV memory: {result.theoretical_kv_mib:.4f} MiB ({result.theoretical_kv_bytes:,} bytes)")
    print(f"KV scaling match: {result.kv_scaling_match_pct:.2f}%")
    print(f"Peak Process RSS: {result.peak_process_rss_mib:.2f} MiB")
    print(f"Peak Device Allocated: {result.peak_device_allocated_mib:.2f} MiB")

    # 3. Context length scaling evaluation
    if args.scaling:
        context_lengths = [512, 1024, 2048, 4096]
        print(f"\n--- Running Context Length Scaling Suite: {context_lengths} ---")
        scaling_results = engine.evaluate_scaling(context_lengths=context_lengths)

        print("\n" + "=" * 70)
        print("CONTEXT LENGTH SCALING SUMMARY")
        print("=" * 70)
        print(f"{'Context (Tokens)':<18}{'Actual KV (MiB)':<18}{'Theo KV (MiB)':<18}{'Prefill (ms)':<16}{'Tokens/sec':<12}")
        print("-" * 70)
        for row in scaling_results:
            print(
                f"{row['actual_input_tokens']:<18}"
                f"{row['kv_cache_mib']:<18.2f}"
                f"{row['theoretical_kv_mib']:<18.2f}"
                f"{row['prefill_time_ms']:<16.1f}"
                f"{row['decode_tokens_per_sec']:<12.1f}"
            )
        print("=" * 70)

        # Save scaling results to disk
        os.makedirs("results", exist_ok=True)
        out_path = "results/baseline_scaling_results.json"
        with open(out_path, "w") as f:
            json.dump({
                "model_meta": meta,
                "verification_result": result.to_dict(),
                "scaling_suite": scaling_results
            }, f, indent=2)
        print(f"Saved baseline scaling results to {out_path}")


if __name__ == "__main__":
    main()
