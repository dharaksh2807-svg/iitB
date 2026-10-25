#!/usr/bin/env python3
"""
Dual-Mode Demonstration Interface for Long-Context KV-Cache Compression.

Usage:
  # Mode A: Interactive CLI Demonstration
  python demo.py --mode interactive
  python demo.py --mode interactive --preset 0 --method ogc_cache

  # Mode B: Terminal Benchmark Dashboard
  python demo.py --mode dashboard
  python demo.py --mode dashboard --phase 8

  # Web Server Mode (Launches Web UI Dashboard & Interactive Interface)
  python demo.py --serve --port 8080
"""

import argparse
import json
import os
import sys
import time

# ANSI Terminal Styling
C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_BLUE = "\033[34m"
C_CYAN = "\033[36m"
C_GREEN = "\033[32m"
C_YELLOW = "\033[33m"
C_RED = "\033[31m"
C_MAGENTA = "\033[35m"
C_BG_DARK = "\033[40m"


def print_banner():
    banner = f"""
{C_BOLD}{C_CYAN}╔═══════════════════════════════════════════════════════════════════════════╗
║   ⚡ KV-CACHE SQUEEZE: LONG-CONTEXT LLM COMPRESSION DEMONSTRATION         ║
║   Hackathon Problem 2: Squeezing Long-Context Inference into Less Memory  ║
╚═══════════════════════════════════════════════════════════════════════════╝{C_RESET}
    """
    print(banner)


def run_interactive_cli(args):
    """Executes Mode A: Interactive Demonstration in terminal."""
    print(f"\n{C_BOLD}{C_GREEN}=== MODE A: INTERACTIVE INFERENCE DEMONSTRATION ==={C_RESET}\n")

    # 1. Load or Select Document
    doc_text = ""
    question_text = ""
    ground_truth = None

    dataset_path = "data/benchmark_dataset.json"
    dataset = []
    if os.path.exists(dataset_path):
        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)

    if args.preset is not None and 0 <= args.preset < len(dataset):
        preset = dataset[args.preset]
        doc_text = preset["document"]
        question_text = preset["question"]
        ground_truth = preset.get("ground_truth_answers")
        print(f"{C_CYAN}Loaded Preset #{args.preset}: {preset['item_id']} "
              f"({preset['target_context_length']} tok | Depth: {preset['depth_category']}){C_RESET}")
    elif args.document:
        if os.path.exists(args.document):
            with open(args.document, "r", encoding="utf-8") as f:
                doc_text = f.read()
            print(f"{C_CYAN}Loaded document from file: {args.document}{C_RESET}")
        else:
            doc_text = args.document
    else:
        # Interactive selection
        print(f"{C_BOLD}Select Input Document Source:{C_RESET}")
        print("  [1] Preset: 1,024 Tokens (Needle at Start, 5% depth)")
        print("  [2] Preset: 1,024 Tokens (Needle in Middle, 50% depth)")
        print("  [3] Preset: 2,048 Tokens (Needle in Early-Middle, 25% depth)")
        print("  [4] Preset: 2,048 Tokens (Needle at End, 95% depth)")
        print("  [5] Preset: 4,096 Tokens (Needle in Middle, 50% depth)")
        print("  [6] Enter / Paste Custom Document Path or Text")

        try:
            choice = input(f"\n{C_YELLOW}Enter choice [1-6] (default: 1): {C_RESET}").strip() or "1"
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            return

        choice_map = {"1": 0, "2": 4, "3": 12, "4": 18, "5": 24}
        if choice in choice_map and choice_map[choice] < len(dataset):
            preset = dataset[choice_map[choice]]
            doc_text = preset["document"]
            question_text = preset["question"]
            ground_truth = preset.get("ground_truth_answers")
            print(f"\n{C_CYAN}Selected: {preset['item_id']}{C_RESET}")
        else:
            path_input = input(f"{C_YELLOW}Enter file path to document: {C_RESET}").strip()
            if os.path.exists(path_input):
                with open(path_input, "r", encoding="utf-8") as f:
                    doc_text = f.read()
            else:
                doc_text = path_input

    # 2. Select Question
    if not question_text:
        if args.question:
            question_text = args.question
        else:
            try:
                question_text = input(f"{C_YELLOW}Enter Question / Inquiry: {C_RESET}").strip()
            except (EOFError, KeyboardInterrupt):
                return

    # 3. Method Selection
    methods = [
        ("full_baseline", "Full Uncompressed Baseline (B = L)", True),
        ("streaming_llm", "StreamingLLM (Attention Sinks + Recent Window)", True),
        ("sliding_window", "Pure Sliding Window (Recent Only, S=0)", True),
        ("h2o", "H2O Heavy-Hitter Cache (Global Attention Oracle)", True),
        ("ogc_cache", "OGC-Cache (Observation-Guided Chunked - Phase 8)", True),
        ("awq_4bit", "4-bit KV Cache Quantization [Unavailable / Requires GPU Kernel]", False),
        ("lococo", "LoCoCo Low-Rank Cache [Unavailable / Not Implemented]", False),
    ]

    selected_method = args.method
    if not selected_method:
        print(f"\n{C_BOLD}Select Inference Method:{C_RESET}")
        for idx, (m_id, m_name, available) in enumerate(methods, 1):
            if available:
                print(f"  [{idx}] {C_GREEN}{m_name}{C_RESET}")
            else:
                print(f"  [{idx}] {C_RED}{m_name}{C_RESET}")

        try:
            m_choice = input(f"\n{C_YELLOW}Enter method [1-5] (default: 5 OGC-Cache): {C_RESET}").strip() or "5"
            idx_int = int(m_choice) - 1
            if 0 <= idx_int < len(methods):
                if not methods[idx_int][2]:
                    print(f"{C_RED}Error: Method '{methods[idx_int][1]}' is not implemented. Please select an active engine.{C_RESET}")
                    return
                selected_method = methods[idx_int][0]
            else:
                selected_method = "ogc_cache"
        except (ValueError, EOFError, KeyboardInterrupt):
            selected_method = "ogc_cache"

    print(f"\n{C_CYAN}Selected Engine: {C_BOLD}{selected_method}{C_RESET}")

    # 4. Budget Configuration
    budget = args.budget or 260
    if selected_method != "full_baseline":
        print(f"Active Cache Budget: {C_BOLD}{budget} tokens{C_RESET} ({budget * 114688 / (1024**2):.2f} MiB physical cache)")

    # 5. Format Prompt
    if question_text and "[Inquiry]" not in doc_text:
        prompt = f"{doc_text}\n\n[Inquiry]\n{question_text}\nAnswer:"
    else:
        prompt = doc_text

    # 6. Execute Real Inference
    print(f"\n{C_YELLOW}⚡ Initializing engine and loading model weights on MPS...{C_RESET}")
    start_time = time.time()

    from src.server import GLOBAL_REGISTRY
    engine = GLOBAL_REGISTRY.get_engine(
        method=selected_method,
        sinks=4,
        window=192 if selected_method == "ogc_cache" else 256,
        chunk_size=64,
        heavy_hitters=budget - 196 if budget > 196 else 64,
    )

    print(f"{C_YELLOW}🚀 Executing prompt prefill and compressed generation...{C_RESET}")
    res = engine.generate(prompt=prompt, max_new_tokens=35)
    total_elapsed = time.time() - start_time

    # 7. Print Formatted Results
    print(f"\n{C_BOLD}{C_GREEN}╔═══════════════════════════════════════════════════════════════════════════╗")
    print(f"║                            INFERENCE RESULT                               ║")
    print(f"╚═══════════════════════════════════════════════════════════════════════════╝{C_RESET}")
    gen_text = getattr(res, "generated_text", getattr(res, "output_text", ""))
    print(f"\n{C_BOLD}Generated Text:{C_RESET}\n{C_CYAN}{gen_text}{C_RESET}\n")

    if ground_truth:
        is_correct = any(gt.lower() in gen_text.lower() for gt in ground_truth)
        status_label = f"{C_GREEN}PASSED ✓{C_RESET}" if is_correct else f"{C_RED}FAILED ✗{C_RESET}"
        print(f"Ground Truth Reference : {ground_truth}")
        print(f"Verification Status    : {status_label}")

    print(f"\n{C_BOLD}{C_MAGENTA}─── PHYSICAL RUNTIME TELEMETRY & MEASUREMENTS ─────────────────────────────{C_RESET}")
    uncomp_mib = res.uncompressed_kv_bytes / (1024 * 1024)
    comp_mib = res.compressed_kv_bytes / (1024 * 1024)
    print(f"Prompt Input Tokens    : {res.input_tokens:,}")
    print(f"Retained Cache Tokens  : {getattr(res, 'retained_tokens', res.input_tokens):,} tokens")
    print(f"Physical KV Cache Size : {C_BOLD}{comp_mib:.2f} MiB{C_RESET} (Uncompressed: {uncomp_mib:.2f} MiB)")
    print(f"KV Memory Reduction    : {C_BOLD}{C_GREEN}{res.kv_cache_savings_pct:.1f}% Savings{C_RESET}")
    print(f"Prefill Latency        : {res.prefill_time_ms:.1f} ms")
    print(f"Decoding Throughput    : {res.decode_tokens_per_sec:.1f} tokens/second")
    print(f"Total Turn Time        : {total_elapsed:.2f} s")
    print(f"{C_MAGENTA}───────────────────────────────────────────────────────────────────────────{C_RESET}\n")


def run_benchmark_dashboard(args):
    """Executes Mode B: Terminal Benchmark Dashboard."""
    print(f"\n{C_BOLD}{C_CYAN}=== MODE B: RESEARCH BENCHMARK DASHBOARD ==={C_RESET}\n")

    p8_file = "results/phase8_ablation_results.json"
    if not os.path.exists(p8_file):
        print(f"{C_RED}Error: {p8_file} not found. Run ablation benchmark first.{C_RESET}")
        return

    with open(p8_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"{C_BOLD}{'Architecture / Configuration':34s} {'Accuracy':12s} {'Delta':14s} {'KV Savings':14s} {'Prefill Latency':16s} {'Decode TPS'}{C_RESET}")
    print("─" * 100)

    base_acc = data["Full Baseline"]["overall_accuracy_pct"]

    for name, info in data.items():
        acc = info["overall_accuracy_pct"]
        diff = acc - base_acc
        diff_str = "Reference" if name == "Full Baseline" else f"{diff:+.2f} pp"
        sav = f"{info['overall_kv_savings_pct']:.2f}%"
        pref = f"{info['avg_prefill_ms']:.1f} ms"
        dec = f"{info['avg_decode_tok_s']:.1f} tok/s"

        if "OGC" in name:
            color = C_GREEN
        elif "Full" in name:
            color = C_BOLD
        else:
            color = C_RESET

        print(f"{color}{name:34s} {acc:6.2f}%     {diff_str:14s} {sav:14s} {pref:16s} {dec}{C_RESET}")

    print("─" * 100)

    print(f"\n{C_BOLD}Granular Depth-Wise Fact Recovery Breakdown (%):{C_RESET}")
    print(f"{'Configuration':34s} {'Start (5%)':12s} {'Early-Mid':12s} {'Middle (50%)':14s} {'Late-Mid':12s} {'End (95%)'}")
    print("─" * 95)
    for name, info in data.items():
        d = info["by_depth"]
        s = f"{d['start']['accuracy_pct']:.1f}%"
        em = f"{d['early_middle']['accuracy_pct']:.1f}%"
        m = f"{d['middle']['accuracy_pct']:.1f}%"
        lm = f"{d['late_middle']['accuracy_pct']:.1f}%"
        e = f"{d['end']['accuracy_pct']:.1f}%"
        print(f"{name:34s} {s:12s} {em:12s} {m:14s} {lm:12s} {e}")
    print("─" * 95)

    print(f"\n{C_CYAN}Generated Comparative Plots Available in `results/`:{C_RESET}")
    print("  • results/plot_phase8_ablation_accuracy.png")
    print("  • results/plot_phase8_depth_recovery.png")
    print("  • results/plot_phase8_pareto_frontier.png")
    print("  • results/plot_phase7_policy_accuracy_comparison.png")

    print(f"\n{C_BOLD}Reproducibility Specifications:{C_RESET}")
    print("  • Hardware : Apple Silicon M5 (Metal Performance Shaders, MPS)")
    print("  • Framework: PyTorch 2.14.1 with Accelerate & Transformers 5.x")
    print("  • Model    : Qwen/Qwen2.5-1.5B-Instruct (Apache 2.0)")
    print("  • Dataset  : data/benchmark_dataset.json (Multi-Depth NIAH, seed=42)")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Dual-Mode Demonstration Interface for Long-Context KV-Cache Compression."
    )
    parser.add_argument(
        "--mode",
        choices=["interactive", "dashboard"],
        default="interactive",
        help="Interface mode: 'interactive' (live inference) or 'dashboard' (benchmark tables).",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Start the local Web UI server and dashboard.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for Web UI server (default: 8080).",
    )
    parser.add_argument(
        "--method",
        choices=["full_baseline", "streaming_llm", "sliding_window", "h2o", "ogc_cache"],
        help="Compression method to run.",
    )
    parser.add_argument(
        "--preset",
        type=int,
        help="Preset benchmark item index (0-29).",
    )
    parser.add_argument(
        "--document",
        type=str,
        help="Path to document file or inline text.",
    )
    parser.add_argument(
        "--question",
        type=str,
        help="Inquiry or question text.",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=260,
        help="KV cache budget in tokens (default: 260).",
    )

    args = parser.parse_args()
    print_banner()

    if args.serve:
        from src.server import run_server
        run_server(port=args.port)
    elif args.mode == "dashboard":
        run_benchmark_dashboard(args)
    else:
        run_interactive_cli(args)


if __name__ == "__main__":
    main()
