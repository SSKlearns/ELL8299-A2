import os
import sys
from transformers import AutoTokenizer
from common import read, write, get_model, load_setup, get_score, adapt_z, get_ents, cond_name, base_name

inp = sys.argv[1]
out = sys.argv[2]

cfg, green = load_setup(out)
kind = cfg["score"]
rows = read(inp)
name = cond_name(inp)

tok = AutoTokenizer.from_pretrained(base_name)
all_ids = [tok(r["text"], add_special_tokens=False)["input_ids"] for r in rows]
ents = None
qs = None
if kind in ["weighted", "mz", "oz", "az", "aoz"]:
    tok, model = get_model(base_name)
    ents, qs = get_ents(model, all_ids, green=green)

res = []
for i in range(len(rows)):
    e = None
    q = None
    if ents is not None:
        e = ents[i]
        q = qs[i]
    if kind == "az" or kind == "aoz":
        s = adapt_z(all_ids[i], green, q, cfg, kind)
    else:
        s = get_score(all_ids[i], green, cfg["gamma"], kind, e, q, cfg["delta"])
    res.append({"id": rows[i]["id"], "score": s})

dest = os.path.join(out, "scores_" + name + ".jsonl")
write(dest, res)
print("wrote", dest, "(" + kind + ")")
