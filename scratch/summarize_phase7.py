import json

with open("results/phase7_comparative_results.json") as f:
    data = json.load(f)

for k, v in data.items():
    print(f"=== {k} ===")
    acc = v["overall_accuracy_pct"]
    corr = v["total_correct"]
    tot = v["total_evaluated"]
    sav = v["overall_kv_savings_pct"]
    pref = v["avg_prefill_ms"]
    dec = v["avg_decode_tok_s"]
    print(f"  Overall Acc: {acc:.2f}% ({corr}/{tot})")
    print(f"  KV Savings: {sav:.2f}%")
    print(f"  Avg Prefill: {pref:.1f} ms | Avg Decode: {dec:.1f} tok/s")
    for d, info in v["by_depth"].items():
        c = info['correct']
        t = info['total']
        p = info['accuracy_pct']
        print(f"    Depth {d:15s}: {c}/{t} ({p:.1f}%)")
    for ctx, info in v["by_context_length"].items():
        c = info['correct']
        t = info['total']
        p = info['accuracy_pct']
        pm = info['avg_prefill_ms']
        sv = info['kv_savings_pct']
        print(f"    Context L={ctx:5s}: {c}/{t} ({p:.1f}%) | Prefill: {pm:.1f} ms | Savings: {sv:.1f}%")
