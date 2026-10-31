"""Investigate BF16 vs FP32 divergence between batched prefill and tokenwise held-query prefill."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.baseline import UncompressedBaselineEngine
from src.ogc_cache import OGCCacheEngine

def test_divergence(dtype_str="bfloat16", device_str="mps"):
    dtype = getattr(torch, dtype_str)
    device = torch.device(device_str)
    print(f"\n=======================================================")
    print(f"Testing with dtype={dtype_str}, device={device_str}")
    print(f"=======================================================")
    
    prompt = "Document: the port is 54321. Question: return the port. Answer:"
    
    base = UncompressedBaselineEngine(device=device_str, dtype=dtype, attn_implementation="eager")
    base.load_model()
    
    inputs = base.tokenizer(prompt, return_tensors="pt").to(device)
    n = inputs.input_ids.shape[1]
    
    # 1. Batched HF generate
    with torch.inference_mode():
        hf_gen = base.model.generate(
            **inputs,
            max_new_tokens=12,
            do_sample=False,
            repetition_penalty=1.0,
            use_cache=True
        ).tolist()[0][n:]
    
    # 2. Held query engine (unpruned)
    engine = OGCCacheEngine(device=device_str, dtype=dtype)
    engine.model, engine.tokenizer = base.model, base.tokenizer
    engine.max_context_length = base.max_context_length
    engine.document_token_count = n // 2
    
    result = engine.generate(prompt, 12)
    held_gen = result.generated_token_ids
    
    print(f"Prompt length: {n} tokens (doc: {n//2}, query: {n - n//2})")
    print(f"HF tokens:   {hf_gen}")
    print(f"Held tokens: {held_gen}")
    print(f"HF text:   '{base.tokenizer.decode(hf_gen)}'")
    print(f"Held text: '{base.tokenizer.decode(held_gen)}'")
    
    # Check token-by-token agreement
    matches = [h == g for h, g in zip(hf_gen, held_gen)]
    print(f"Token matches: {matches}")
    divergence_idx = next((i for i, m in enumerate(matches) if not m), None)
    if divergence_idx is None:
        print("ALL 12 TOKENS MATCH EXACTLY!")
    else:
        print(f"DIVERGENCE at token index {divergence_idx}: HF={hf_gen[divergence_idx]} ('{base.tokenizer.decode([hf_gen[divergence_idx]])}') vs Held={held_gen[divergence_idx]} ('{base.tokenizer.decode([held_gen[divergence_idx]])}')")

    # Now let's trace logits at step 0 (first answer token) and subsequent steps if forced with identical prefixes
    print("\nTracing step-by-step logits when conditioned on identical prefix:")
    # We will step through tokens 0 to 11 using teacher-forced ground truth (hf_gen) prefix to compare logits
    # Compare forward pass on full prefix vs tokenwise forward pass
    current_ids = inputs.input_ids
    past_kv_hf = None
    with torch.inference_mode():
        out_hf = base.model(input_ids=current_ids, use_cache=True)
        logits_hf_step0 = out_hf.logits[0, -1, :].float()
        past_kv_hf = out_hf.past_key_values
    
    # Now simulate tokenwise query feeding (like held_query)
    doc_ids = inputs.input_ids[:, :n//2]
    query_ids = inputs.input_ids[:, n//2:]
    with torch.inference_mode():
        out_doc = base.model(input_ids=doc_ids, use_cache=True)
        past_kv_held = out_doc.past_key_values
        for i in range(query_ids.shape[1]):
            q_token = query_ids[:, i:i+1]
            out_q = base.model(input_ids=q_token, past_key_values=past_kv_held, use_cache=True)
            past_kv_held = out_q.past_key_values
        logits_held_step0 = out_q.logits[0, -1, :].float()
    
    diff_step0 = (logits_hf_step0 - logits_held_step0).abs()
    print(f"Step 0 logit diff: max={diff_step0.max().item():.6f}, mean={diff_step0.mean().item():.6f}")
    top2_hf = torch.topk(logits_hf_step0, 2)
    top2_held = torch.topk(logits_held_step0, 2)
    print(f"Step 0 HF top 2: {top2_hf.indices.tolist()} with logits {top2_hf.values.tolist()}, margin = {top2_hf.values[0] - top2_hf.values[1]:.4f}")
    print(f"Step 0 Held top 2: {top2_held.indices.tolist()} with logits {top2_held.values.tolist()}, margin = {top2_held.values[0] - top2_held.values[1]:.4f}")

    # Now step through following identical tokens:
    kv_hf_step = past_kv_hf
    kv_held_step = past_kv_held
    for step in range(len(hf_gen)):
        tok = hf_gen[step]
        tok_tensor = torch.tensor([[tok]], device=device)
        with torch.inference_mode():
            out_hf_next = base.model(input_ids=tok_tensor, past_key_values=kv_hf_step, use_cache=True)
            kv_hf_step = out_hf_next.past_key_values
            logits_hf = out_hf_next.logits[0, -1, :].float()
            
            out_held_next = base.model(input_ids=tok_tensor, past_key_values=kv_held_step, use_cache=True)
            kv_held_step = out_held_next.past_key_values
            logits_held = out_held_next.logits[0, -1, :].float()
        
        diff = (logits_hf - logits_held).abs()
        max_d = diff.max().item()
        mean_d = diff.mean().item()
        top_hf = torch.argmax(logits_hf).item()
        top_held = torch.argmax(logits_held).item()
        top2_h = torch.topk(logits_hf, 2)
        margin = (top2_h.values[0] - top2_h.values[1]).item()
        
        status = "MATCH" if top_hf == top_held else f"FLIP (HF={top_hf} vs Held={top_held})"
        print(f"Step {step+1} (after token '{base.tokenizer.decode([tok])}'): max_diff={max_d:.6f}, margin={margin:.6f} -> {status}")
        if top_hf != top_held:
            print(f"   HF top-2: {top2_h.indices.tolist()} ({[base.tokenizer.decode([idx]) for idx in top2_h.indices.tolist()]})")
            top2_g = torch.topk(logits_held, 2)
            print(f"   Held top-2: {top2_g.indices.tolist()} ({[base.tokenizer.decode([idx]) for idx in top2_g.indices.tolist()]})")

if __name__ == "__main__":
    test_divergence("bfloat16", "mps")
    test_divergence("float32", "cpu")

