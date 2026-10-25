import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json

with open("data/benchmark_dataset.json") as f:
    items = json.load(f)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct",
    dtype=torch.bfloat16,
    attn_implementation="eager",
    device_map=None
)
model.to("mps")
model.eval()

# Test items across different depths
test_indices = [0, 1, 2, 4, 8] # start, start multi, early_middle, middle, end
for idx in test_indices:
    item = items[idx]
    input_ids = tokenizer(item["document"], return_tensors="pt")["input_ids"].to("mps")
    L = input_ids.shape[1]

    with torch.no_grad():
        outputs = model(input_ids, use_cache=True, output_attentions=True)

    past_key_values = outputs.past_key_values

    # Query window attention (last 48 tokens)
    W_query = 48
    query_start = max(0, L - W_query)
    query_scores = torch.zeros(L, device="mps", dtype=torch.float32)

    for layer_attn in outputs.attentions:
        head_sum = layer_attn[0].sum(dim=0).to(torch.float32)
        query_scores += head_sum[query_start:].sum(dim=0)

    del outputs
    torch.mps.empty_cache()

    S = 4
    W = 192
    C = 64
    num_chunks = 1

    sink_indices = list(range(S))
    recent_indices = list(range(max(0, L - W), L))

    # Slide window across candidate range
    cand_start = S
    cand_end = max(cand_start, L - W)
    if cand_end - cand_start > C:
        best_score = -1e9
        best_start = cand_start
        for s_idx in range(cand_start, cand_end - C + 1):
            sc = query_scores[s_idx : s_idx + C].sum().item()
            if sc > best_score:
                best_score = sc
                best_start = s_idx
        chunk_indices = list(range(best_start, best_start + C))
    else:
        chunk_indices = []

    retained = sorted(list(set(sink_indices + chunk_indices + recent_indices)))
    retained_tensor = torch.tensor(retained, device="mps")

    for layer in past_key_values.layers:
        if layer.keys is not None and layer.values is not None:
            layer.keys = layer.keys[..., retained_tensor, :].contiguous()
            layer.values = layer.values[..., retained_tensor, :].contiguous()

    cur_token = input_ids[:, -1:]
    generated_tokens = []
    for step in range(25):
        pos_ids = torch.tensor([[L + step]], dtype=torch.long, device="mps")
        with torch.no_grad():
            out = model(
                input_ids=cur_token,
                past_key_values=past_key_values,
                position_ids=pos_ids,
                use_cache=True
            )
        logits = out.logits[:, -1, :]
        next_tok = torch.argmax(logits, dim=-1, keepdim=True)
        generated_tokens.append(next_tok.item())
        cur_token = next_tok
        decoded_so_far = tokenizer.decode(generated_tokens)
        if next_tok.item() == tokenizer.eos_token_id or '\n' in decoded_so_far:
            break

    answer = tokenizer.decode(generated_tokens).strip()
    gts = item['ground_truth_answers']
    correct = all(gt in answer for gt in gts)
    print(f"[{item['item_id']}] Depth: {item['depth_category']} | Retained: {len(retained)}/{L} | Ans: {repr(answer)} | GT: {gts} | Passed: {correct}")
