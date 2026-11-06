import sys
import numpy as np
from common import read, write, get_model, bias_prompt, qa_prompt, get_scores

mode = sys.argv[1]
inp = sys.argv[2]
dest = sys.argv[3]

rows = read(inp)
tok, model = get_model()
if mode == "qa":
    prompts = [qa_prompt(tok, r) for r in rows]
    bs = 8
else:
    prompts = [bias_prompt(tok, r) for r in rows]
    bs = 16
scores = get_scores(tok, model, prompts, [len(r["options"]) for r in rows], bs=bs)
res = []
for i in range(len(rows)):
    res.append({"id": rows[i]["id"], "pred_idx": int(np.argmax(scores[i]))})
write(dest, res)
print("wrote", dest)
