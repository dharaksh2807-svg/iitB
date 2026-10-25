import json

with open("results/phase8_ablation_results.json") as f:
    d = json.load(f)

for name in [
    "Full Baseline",
    "StreamingLLM (S=4, W=256)",
    "Pure Sliding Window (W=260)",
    "H2O Heavy-Hitter (H=64, W=196)",
    "OGC-Cache (K=1, C=64)",
    "OGC-Cache (K=2, C=32)",
]:
    info = d[name]
    acc = info['overall_accuracy_pct']
    corr = info['total_correct']
    tot = info['total_evaluated']
    pref = info['avg_prefill_ms']
    dec = info['avg_decode_tok_s']
    sav = info['overall_kv_savings_pct']
    print(f"=== {name} ===")
    print(f"  Overall: {acc:.2f}% ({corr}/{tot}) | Savings: {sav:.2f}% | Prefill: {pref:.1f}ms | Decode: {dec:.1f} tok/s")
    for depth, d_info in info['by_depth'].items():
        c = d_info['correct']
        t = d_info['total']
        p = d_info['accuracy_pct']
        print(f"    {depth:15s}: {c}/{t} ({p:.1f}%)")
    for ctx, c_info in info['by_context_length'].items():
        c = c_info['correct']
        t = c_info['total']
        p = c_info['accuracy_pct']
        pm = c_info['avg_prefill_ms']
        print(f"    L={ctx:5s}: {c}/{t} ({p:.1f}%) | Prefill: {pm:.1f}ms")
