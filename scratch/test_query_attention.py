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
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",
    device_map="mps"
)

input_ids = tokenizer(item["document"], return_tensors="pt")["input_ids"].to("mps")
L = input_ids.shape[1]
print(f"Sequence length L={L}")

# Run prefill with output_attentions=True
with torch.no_grad():
    outputs = model(input_ids, output_attentions=True)

# Find where needle tokens are
needle_sentence = item['needles'][0]['sentence']
needle_ids = tokenizer(needle_sentence, add_special_tokens=False)["input_ids"]
print(f"Needle sentence: {needle_sentence}")
print(f"Needle token IDs: {needle_ids[:5]}...")

# Search for needle token positions in input_ids
input_list = input_ids[0].tolist()
needle_start_idx = -1
for i in range(len(input_list) - len(needle_ids) + 1):
    if input_list[i:i+len(needle_ids)] == needle_ids:
        needle_start_idx = i
        break
needle_indices = set(range(needle_start_idx, needle_start_idx + len(needle_ids)))
print(f"Needle is at token indices {needle_start_idx} to {needle_start_idx + len(needle_ids)}")

# Compare Global Attention sum vs Query-Window Attention sum
# Shape of each layer attention: [1, num_heads, L, L]
# Global attention sum:
global_scores = torch.zeros(L, device="mps", dtype=torch.float32)
# Query attention sum (last 32 tokens):
query_scores = torch.zeros(L, device="mps", dtype=torch.float32)

W_query = 32
query_start = L - W_query

for layer_attn in outputs.attentions:
    # layer_attn: [1, 12, L, L]
    # sum over heads: [L, L]
    head_sum = layer_attn[0].sum(dim=0).to(torch.float32)
    # Global sum: sum over all queries q
    global_scores += head_sum.sum(dim=0)
    # Query window sum: sum over only q in [query_start, L]
    query_scores += head_sum[query_start:].sum(dim=0)

# Check top 64 tokens from global vs query
top64_global = set(torch.topk(global_scores[:L-196], k=64).indices.tolist())
top64_query = set(torch.topk(query_scores[:L-196], k=64).indices.tolist())

global_needle_hits = len(needle_indices.intersection(top64_global))
query_needle_hits = len(needle_indices.intersection(top64_query))

print(f"\n--- COMPARISON ---")
print(f"Needle indices span: {needle_start_idx}..{needle_start_idx+len(needle_ids)} ({len(needle_indices)} tokens)")
print(f"Global H2O top-64 contains needle tokens: {global_needle_hits} / {len(needle_indices)}")
print(f"Query-Window top-64 contains needle tokens: {query_needle_hits} / {len(needle_indices)}")

# Check top 10 tokens from each
print("\nTop 10 Global Tokens:")
for idx in torch.topk(global_scores[:L-196], k=10).indices.tolist():
    tok = tokenizer.decode([input_list[idx]])
    in_needle = " [NEEDLE!]" if idx in needle_indices else ""
    print(f"  idx {idx:4d}: {repr(tok)}{in_needle}")

print("\nTop 10 Query-Window Tokens:")
for idx in torch.topk(query_scores[:L-196], k=10).indices.tolist():
    tok = tokenizer.decode([input_list[idx]])
    in_needle = " [NEEDLE!]" if idx in needle_indices else ""
    print(f"  idx {idx:4d}: {repr(tok)}{in_needle}")
