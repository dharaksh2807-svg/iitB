"""Build a rigorous, held-out evaluation dataset for Phase 4 validation."""
import json
import random
import hashlib
from transformers import AutoTokenizer
from src.dataset import generate_distractor_paragraph

def build_phase4_suite(seed=2026110801):
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    rng = random.Random(seed)
    
    services = [
        ("Authentication Gateway", "auth-gateway.internal.net", "eu-central-1"),
        ("Billing Dispatcher", "billing-dispatcher.prod.corp", "us-east-2"),
        ("Telemetry Aggregator", "telemetry-aggregator.infra.io", "ap-southeast-1"),
        ("Payment Clearinghouse", "payment-clearinghouse.fintech.int", "us-west-1"),
        ("Edge Routing Proxy", "edge-router.edge.mesh", "sa-east-1"),
    ]
    
    depths = [
        ("start", 0.05),
        ("middle", 0.50),
        ("end", 0.90),
    ]
    
    cases = []
    case_idx = 0
    
    # 1. Synthetic adversarial & multi-fact cases across 1024, 2048, 4096
    for length in [1024, 2048, 4096]:
        for dname, dfrac in depths:
            # Adversarial synthetic case
            cseed = seed + case_idx * 31
            crng = random.Random(cseed)
            proj = f"PROJ-{crng.getrandbits(48):012X}"
            cur_port = str(crng.randrange(10000, 60000))
            obs_port = str(crng.randrange(60001, 65535))
            
            preamble = "Read the records below. Use CURRENT records only; ignore ARCHIVED records.\n\n"
            fact = f"\n\nCURRENT record for project {proj}: primary database port is {cur_port}.\n\n"
            obs = f"\n\nARCHIVED obsolete record for project {proj}: primary database port is {obs_port}.\n"
            question = f"For project {proj}, return only the current primary database port."
            inquiry = f"\n[Inquiry]\n{question}\nAnswer:"
            
            # Assemble with distractors
            case = assemble_document(tok, crng, length, dfrac, preamble, [(dfrac, fact)], obs + inquiry, question, [cur_port])
            case["item_id"] = f"{seed}/{length}/{dname}/adversarial/0"
            case["depth_category"] = dname
            case["depth_fraction"] = dfrac
            case["query_type"] = "adversarial"
            case["obsolete_answer"] = obs_port
            cases.append(case)
            case_idx += 1
            
            # Multi-fact synthetic case
            if length in [1024, 2048]:
                cseed = seed + case_idx * 31
                crng = random.Random(cseed)
                proj = f"PROJ-{crng.getrandbits(48):012X}"
                port = str(crng.randrange(10000, 60000))
                region = crng.choice(["eu-west-1", "ap-south-2", "us-east-1", "sa-east-1"])
                fact1 = f"\n\nCURRENT record for project {proj}: primary database port is {port}.\n\n"
                fact2 = f"\n\nCURRENT record for project {proj}: assigned deployment region is {region}.\n\n"
                f2_frac = (dfrac + 0.35) % 0.85 + 0.05
                question = f"For project {proj}, return only the current primary database port and deployment region, separated by a comma."
                inquiry = f"\n[Inquiry]\n{question}\nAnswer:"
                facts = sorted([(dfrac, fact1), (f2_frac, fact2)])
                case = assemble_document(tok, crng, length, dfrac, preamble, facts, inquiry, question, [port, region])
                case["item_id"] = f"{seed}/{length}/{dname}/multi_fact/0"
                case["depth_category"] = dname
                case["depth_fraction"] = dfrac
                case["query_type"] = "multi_fact"
                cases.append(case)
                case_idx += 1

    # 2. Natural language cases (non-synthetic identifiers)
    for idx, (svc_name, host, region) in enumerate(services):
        length = 2048 if idx % 2 == 0 else 1024
        dname, dfrac = depths[idx % len(depths)]
        cseed = seed + 500 + idx * 47
        crng = random.Random(cseed)
        port = str(crng.randrange(10000, 60000))
        obs_port = str(crng.randrange(60001, 65535))
        
        preamble = "Infrastructure Specification and Service Registry.\nUse CURRENT active service parameters only; ignore obsolete or retired versions.\n\n"
        fact = f"\n\nCURRENT active deployment for {svc_name} (host {host}): ingress listening port is {port}.\n\n"
        obs = f"\n\nARCHIVED legacy configuration for {svc_name}: ingress listening port is {obs_port}.\n"
        question = f"What is the current ingress listening port for the {svc_name}?"
        inquiry = f"\n[Inquiry]\n{question}\nAnswer:"
        
        case = assemble_document(tok, crng, length, dfrac, preamble, [(dfrac, fact)], obs + inquiry, question, [port])
        case["item_id"] = f"{seed}/{length}/{dname}/natural_language/{idx}"
        case["depth_category"] = dname
        case["depth_fraction"] = dfrac
        case["query_type"] = "adversarial"  # It is an adversarial test with natural language!
        case["obsolete_answer"] = obs_port
        cases.append(case)

    print(f"Total cases generated: {len(cases)}")
    return cases

def assemble_document(tok, rng, target_length, depth_frac, preamble, facts, suffix, question, ground_truths):
    overhead = len(tok.encode(preamble + suffix + "".join(f for _, f in facts), add_special_tokens=False))
    filler_target = max(64, target_length - overhead)
    filler_ids = []
    while len(filler_ids) < filler_target + 128:
        filler_ids.extend(tok.encode(generate_distractor_paragraph(rng) + "\n\n", add_special_tokens=False))
        
    def assemble(count):
        parts, cursor = [preamble], 0
        for frac, fact_text in facts:
            boundary = int(count * frac)
            parts.append(tok.decode(filler_ids[cursor:boundary], skip_special_tokens=True))
            parts.append(fact_text)
            cursor = boundary
        parts.append(tok.decode(filler_ids[cursor:count], skip_special_tokens=True))
        text = "".join(parts) + suffix
        if hasattr(tok, "apply_chat_template"):
            text = tok.apply_chat_template([{"role": "user", "content": text}],
                                           tokenize=False, add_generation_prompt=True)
        return text

    for _ in range(6):
        doc = assemble(filler_target)
        actual = len(tok.encode(doc))
        if abs(actual - target_length) <= 2:
            break
        filler_target += target_length - actual
        
    # Find explicit inquiry boundary
    pos = doc.rfind("[Inquiry]")
    offsets = tok(doc, return_offsets_mapping=True)["offset_mapping"]
    doc_token_count = next(i for i, (_, end) in enumerate(offsets) if end > pos)
    
    return {
        "target_context_length": target_length,
        "actual_token_length": actual,
        "document": doc,
        "question": question,
        "ground_truth_answers": ground_truths,
        "document_token_count": doc_token_count,
        "seed": rng.randint(0, 2**32 - 1)
    }

if __name__ == "__main__":
    cases = build_phase4_suite()
    out_path = "artifacts/phase4-fresh-eval/heldout_cases.json"
    import os
    os.makedirs("artifacts/phase4-fresh-eval", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(cases, f, indent=2)
    print(f"Saved {len(cases)} cases to {out_path}")

