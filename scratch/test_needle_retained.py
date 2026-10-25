import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json

with open("data/benchmark_dataset.json") as f:
    items = json.load(f)

item = items[0]
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
input_ids = tokenizer(item["document"], return_tensors="pt")["input_ids"]
input_list = input_ids[0].tolist()

# Find needle
needle_sentence = item['needles'][0]['sentence']
needle_ids = tokenizer(needle_sentence, add_special_tokens=False)["input_ids"]
needle_start_idx = -1
for i in range(len(input_list) - len(needle_ids) + 1):
    if input_list[i:i+len(needle_ids)] == needle_ids:
        needle_start_idx = i
        break

print(f"Needle spans tokens {needle_start_idx} to {needle_start_idx + len(needle_ids)}")
for idx in range(needle_start_idx, needle_start_idx + len(needle_ids)):
    print(f"  Token {idx:3d}: {repr(tokenizer.decode([input_list[idx]]))}")
