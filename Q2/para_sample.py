import os
import sys
import torch
from common import read, write, get_model, inst_name

inp = sys.argv[1]
out = sys.argv[2]
seed = int(sys.argv[3])
name = sys.argv[4] if len(sys.argv) > 4 else "parasample"
bs = 16

rows = read(inp)
tok, model = get_model(inst_name)
torch.manual_seed(seed)
prompts = []
for r in rows:
    text = "Paraphrase the following text. Keep the same meaning and output only the paraphrased text.\n\n" + r["text"]
    prompts.append(tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True))
res = []
for i in range(0, len(rows), bs):
    batch = prompts[i:i + bs]
    enc = tok(batch, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    with torch.no_grad():
        o = model.generate(**enc, max_new_tokens=320, pad_token_id=tok.pad_token_id)
    texts = tok.batch_decode(o[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    for j in range(len(batch)):
        res.append({"id": rows[i + j]["id"], "text": texts[j].strip()})
write(os.path.join(out, "generated_" + name + "_" + str(seed) + ".jsonl"), res)
print("wrote sampled paraphrases (model default generation config)", len(res))
