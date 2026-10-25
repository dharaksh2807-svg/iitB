import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json

with open("data/benchmark_dataset.json") as f:
    items = json.load(f)

item = items[0] # item_L1024_start_single_fact_001, depth=0.05
print(f"Testing item: {item['item_id']}, needle: {item['needles'][0]['key_value']}")

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct",
    dtype=torch.bfloat16,
    attn_implementation="eager",
    device_map=None
)
model.to("mps")
model.eval()

input_ids = tokenizer(item["document"], return_tensors="pt")["input_ids"].to("mps")
L = input_ids.shape[1]

with torch.no_grad():
    outputs = model(input_ids, use_cache=True, output_attentions=True)

past_key_values = outputs.past_key_values

# Query window attention (last 48 tokens)
W_query = 48
query_start = L - W_query
query_scores = torch.zeros(L, device="mps", dtype=torch.float32)

for layer_attn in outputs.attentions:
    head_sum = layer_attn[0].sum(dim=0).to(torch.float32)
    query_scores += head_sum[query_start:].sum(dim=0)

del outputs
torch.mps.empty_cache()

# Budget B = 260:
# S = 4 sink tokens
# W = 192 recent window tokens
# That leaves 64 tokens for Historical Chunks!
# E.g. Two contiguous chunks of size C=32 tokens, or one chunk of 64 tokens!
S = 4
W = 192
C = 64  # single chunk of 64 tokens
num_chunks = 1 # 1 * 64 = 64 tokens

sink_indices = list(range(S))
recent_indices = list(range(L - W, L))
protected = set(sink_indices + recent_indices)

# Compute sliding window sum over candidate range [S, L - W - C]
cand_start = S
cand_end = L - W
window_scores = []
for start_idx in range(cand_start, cand_end - C + 1):
    chunk_score = query_scores[start_idx : start_idx + C].sum().item()
    window_scores.append((chunk_score, start_idx))

# Sort chunks by score descending, ensuring non-overlapping
window_scores.sort(key=lambda x: x[0], reverse=True)
selected_chunk_starts = []
for score, start_idx in window_scores:
    # Check overlap
    overlap = False
    for prev_start in selected_chunk_starts:
        if abs(start_idx - prev_start) < C:
            overlap = True
            break
    if not overlap:
        selected_chunk_starts.append(start_idx)
        if len(selected_chunk_starts) == num_chunks:
            break

chunk_indices = []
for s_idx in selected_chunk_starts:
    chunk_indices.extend(range(s_idx, s_idx + C))
    chunk_text = tokenizer.decode(input_ids[0, s_idx : s_idx + C])
    print(f"Selected chunk at {s_idx}..{s_idx+C}: {repr(chunk_text[:60])}...")

retained_indices = torch.tensor(sorted(sink_indices + chunk_indices + recent_indices), device="mps")
print(f"Total retained tokens: {len(retained_indices)} / {L}")

# Physical slicing across all layers
for layer in past_key_values.layers:
    if layer.keys is not None and layer.values is not None:
        layer.keys = layer.keys[..., retained_indices, :].contiguous()
        layer.values = layer.values[..., retained_indices, :].contiguous()

# Autoregressive generation
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
print(f"\nPrediction: {repr(answer)}")
print(f"Ground truth: {item['ground_truth_answers']}")
print(f"Is Correct: {item['ground_truth_answers'][0] in answer}")
