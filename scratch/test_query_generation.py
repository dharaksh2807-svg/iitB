import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json

with open("data/benchmark_dataset.json") as f:
    items = json.load(f)

item = items[0] # item_L1024_start_single_fact_001, depth=0.05 (near beginning)
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

# 1. Prefill
with torch.no_grad():
    outputs = model(input_ids, use_cache=True, output_attentions=True)

past_key_values = outputs.past_key_values

# Query window attention (last 48 tokens, covering the question prompt)
W_query = 48
query_start = L - W_query
query_scores = torch.zeros(L, device="mps", dtype=torch.float32)

for layer_attn in outputs.attentions:
    head_sum = layer_attn[0].sum(dim=0).to(torch.float32)
    query_scores += head_sum[query_start:].sum(dim=0)

del outputs
torch.mps.empty_cache()

# Budget B = 260: S=4 sinks, W=196 recent, H=60 query-directed heavy hitters
S = 4
W = 196
H = 60

sink_indices = list(range(S))
recent_indices = list(range(L - W, L))
protected = set(sink_indices + recent_indices)

candidate_indices = [i for i in range(L) if i not in protected]
cand_scores = query_scores[candidate_indices]
top_h_rel = torch.topk(cand_scores, k=min(H, len(candidate_indices))).indices.tolist()
hh_indices = [candidate_indices[i] for i in top_h_rel]

retained_indices = torch.tensor(sorted(sink_indices + hh_indices + recent_indices), device="mps")
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
print(f"Prediction: {repr(answer)}")
print(f"Ground truth: {item['ground_truth_answers']}")
print(f"Is Correct: {item['ground_truth_answers'][0] in answer}")
