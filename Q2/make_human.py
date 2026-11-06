import os
import sys
from transformers import AutoTokenizer
from common import read, write, base_name

inp = sys.argv[1]
out = sys.argv[2]

rows = read(inp)
tok = AutoTokenizer.from_pretrained(base_name)
full = []
short = []
for r in rows:
    ids = tok(r["human_text"], add_special_tokens=False)["input_ids"]
    full.append({"id": r["id"], "text": r["human_text"]})
    short.append({"id": r["id"], "text": tok.decode(ids[:50], skip_special_tokens=True)})
write(os.path.join(out, "human.jsonl"), full)
write(os.path.join(out, "human_short.jsonl"), short)
print("wrote human.jsonl and human_short.jsonl with", len(full), "texts")
