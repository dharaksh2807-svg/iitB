"""Deep adversarial regression analysis joining dataset.json and outputs.jsonl."""
import json
import re
from collections import defaultdict

def run():
    base_dir = "artifacts/phase3-reserved-held-60"
    with open(f"{base_dir}/dataset.json") as f:
        dataset = {it["item_id"]: it for it in json.load(f) if it["query_type"] == "adversarial"}
        
    with open(f"{base_dir}/outputs.jsonl") as f:
        outputs = [json.loads(line) for line in f if json.loads(line).get("query_type") == "adversarial"]

    print(f"Loaded {len(dataset)} dataset items and {len(outputs)} output records.")
    
    # Map config_name to clean policy names
    def clean_policy(cfg):
        if cfg.startswith("entity_reserved"): return "entity_reserved"
        if cfg.startswith("entity_balanced"): return "entity_balanced"
        if cfg.startswith("ogc"): return "ogc"
        if cfg == "full": return "full"
        return cfg

    by_case = defaultdict(dict)
    for r in outputs:
        pol = clean_policy(r["config_name"])
        by_case[r["item_id"]][pol] = r

    policies = ["full", "ogc", "entity_balanced", "entity_reserved"]
    
    # Regex to find obsolete value
    obs_re = re.compile(r"ARCHIVED obsolete record for project [^:]+: primary database port is (\d+)")
    cur_re = re.compile(r"CURRENT record for project [^:]+: primary database port is (\d+)")

    print("\n" + "="*90)
    print("DETAILED CASE ANALYSIS: ALL 20 ADVERSARIAL CASES")
    print("="*90)

    summary = defaultdict(lambda: defaultdict(int))
    by_depth = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    for item_id, item in sorted(dataset.items(), key=lambda x: (x[1]["target_context_length"], x[1]["depth_category"], x[0])):
        doc = item["document"]
        m_obs = obs_re.search(doc)
        obs_val = m_obs.group(1) if m_obs else None
        m_cur = cur_re.search(doc)
        cur_val = item["ground_truth_answers"][0]
        depth = item["depth_category"]
        length = item["target_context_length"]
        
        print(f"\nItem: {item_id} | Length: {length} | Depth: {depth}")
        print(f"Current Target: {cur_val} | Obsolete Target: {obs_val}")
        
        # Check where current and obsolete records appear in character/token space
        cur_span = (m_cur.start(), m_cur.end()) if m_cur else (0, 0)
        obs_span = (m_obs.start(), m_obs.end()) if m_obs else (0, 0)
        print(f"Current char span: {cur_span} | Obsolete char span: {obs_span} | Total chars: {len(doc)}")
        
        for pol in policies:
            r = by_case[item_id].get(pol)
            if not r:
                print(f"  [{pol:15s}] MISSING")
                continue
            pred = str(r["prediction"]).strip()
            correct = r["is_correct"]
            
            # Categorize failure
            if correct:
                cat = "CORRECT"
            elif pred == obs_val or (obs_val and obs_val in pred):
                cat = "OBEYED_OBSOLETE"
            elif cur_val in pred:
                cat = "PARTIAL/FORMAT"
            else:
                cat = "OTHER/HALLUCINATED"
                
            summary[pol][cat] += 1
            by_depth[depth][pol][cat] += 1
            
            # Check positions retained
            doc_retained = set(r.get("document_retained_positions", []))
            final_retained = set(r.get("retained_positions", []))
            
            print(f"  [{pol:15s}] Corr: {str(correct):5s} | Cat: {cat:17s} | Pred: '{pred[:30]}' | RetainedTokens: {len(final_retained)}")

    print("\n" + "="*90)
    print("OVERALL SUMMARY BY POLICY")
    print("="*90)
    for pol in policies:
        corr = summary[pol]['CORRECT']
        obs = summary[pol]['OBEYED_OBSOLETE']
        oth = summary[pol]['OTHER/HALLUCINATED']
        part = summary[pol]['PARTIAL/FORMAT']
        print(f"Policy: {pol:15s} | Correct: {corr:2d}/20 ({corr*5:2d}%) | Obeyed Obsolete: {obs:2d} | Hallucinated: {oth:2d} | Format: {part:2d}")

    print("\n" + "="*90)
    print("BREAKDOWN BY DEPTH CATEGORY (EACH DEPTH HAS 4 CASES: 2 at 1024, 2 at 2048)")
    print("="*90)
    for depth in ["start", "early_middle", "middle", "late_middle", "end"]:
        print(f"\n--- Depth: {depth.upper()} (4 cases) ---")
        for pol in policies:
            corr = by_depth[depth][pol]['CORRECT']
            obs = by_depth[depth][pol]['OBEYED_OBSOLETE']
            oth = by_depth[depth][pol]['OTHER/HALLUCINATED']
            print(f"  {pol:15s} -> Correct: {corr}/4 | Obeyed Obsolete: {obs}/4 | Other: {oth}/4")

if __name__ == "__main__":
    run()
