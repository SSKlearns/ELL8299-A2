import os
import sys
import random
import torch
from transformers import AutoTokenizer
from common import read, write, get_model, base_name, inst_name

inp = sys.argv[1]
out = sys.argv[2]
seed = int(sys.argv[3])
bs = 16

rows = read(inp)
tok = AutoTokenizer.from_pretrained(base_name)
n_real = tok.convert_tokens_to_ids("<|endoftext|>")
rng = random.Random(seed)

clean = []
short = []
sub = []
for r in rows:
    ids = r["token_ids"]
    clean.append({"id": r["id"], "text": r["text"]})
    short.append({"id": r["id"], "text": tok.decode(ids[:50], skip_special_tokens=True)})
    new = list(ids)
    for p in rng.sample(range(len(ids)), int(0.2 * len(ids))):
        new[p] = rng.randrange(n_real)
    sub.append({"id": r["id"], "text": tok.decode(new, skip_special_tokens=True)})

tok2, model2 = get_model(inst_name)
prompts = []
for r in rows:
    text = "Paraphrase the following text. Keep the same meaning and output only the paraphrased text.\n\n" + r["text"]
    prompts.append(tok2.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True))
para = []
for i in range(0, len(rows), bs):
    batch = prompts[i:i + bs]
    enc = tok2(batch, return_tensors="pt", padding=True, add_special_tokens=False).to(model2.device)
    with torch.no_grad():
        o = model2.generate(**enc, max_new_tokens=320, do_sample=False, pad_token_id=tok2.pad_token_id)
    texts = tok2.batch_decode(o[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    for j in range(len(batch)):
        para.append({"id": rows[i + j]["id"], "text": texts[j].strip()})
    print("paraphrased", len(para), "/", len(rows))

write(os.path.join(out, "generated_clean_" + str(seed) + ".jsonl"), clean)
write(os.path.join(out, "generated_short_" + str(seed) + ".jsonl"), short)
write(os.path.join(out, "generated_substitution_" + str(seed) + ".jsonl"), sub)
write(os.path.join(out, "generated_paraphrase_" + str(seed) + ".jsonl"), para)
print("wrote 4 perturbed files to", out)
