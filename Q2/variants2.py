import os
import sys
import math
import numpy as np
from transformers import AutoTokenizer
from common import read, get_model, load_setup, get_score, adapt_z, get_ents, base_name

folder = sys.argv[1]
seed = sys.argv[2]
cfg, green = load_setup(folder)
tok, model = get_model(base_name)
files = {"human": "human.jsonl", "human_short": "human_short.jsonl"}
for c in ["clean", "short", "substitution", "paraphrase", "parasample"]:
    files[c] = "generated_%s_%s.jsonl" % (c, seed)
kinds = ["mz", "oz"]
if "q0" in cfg:
    kinds = kinds + ["az", "aoz"]
sc = {}
for c in files:
    rows = read(os.path.join(folder, files[c]))
    ids = [tok(r["text"], add_special_tokens=False)["input_ids"] for r in rows]
    ents, qs = get_ents(model, ids, green=green)
    for k in kinds:
        v = []
        for i in range(len(ids)):
            if k in ["az", "aoz"]:
                v.append(adapt_z(ids[i], green, qs[i], cfg, k))
            else:
                v.append(get_score(ids[i], green, cfg["gamma"], k, ents[i], qs[i], cfg.get("hi", cfg["delta"])))
        sc[(c, k)] = np.array(v)
for k in kinds:
    line = "%-4s" % k
    te = []
    tg = []
    for c in ["clean", "short", "substitution", "paraphrase", "parasample"]:
        n = sc[("human_short" if c == "short" else "human", k)]
        tau = np.sort(n)[math.ceil(0.99 * len(n)) - 1]
        tau_g = n.mean() + 2.3263 * n.std()
        e = float((sc[(c, k)] > tau).mean())
        g = float((sc[(c, k)] > tau_g).mean())
        te.append(e)
        tg.append(g)
        line += " | %s %.2f/%.2f" % (c[:5], e, g)
    line += " || S_det(para) %.4f/%.4f  S_det(sampled para) %.4f/%.4f" % (np.mean(te[:4]), np.mean(tg[:4]), np.mean(te[:3] + te[4:]), np.mean(tg[:3] + tg[4:]))
    print(line)
print("(each cell: TPR with the spec threshold / with the Gaussian threshold)")
