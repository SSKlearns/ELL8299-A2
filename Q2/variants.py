import os
import sys
import math
from transformers import AutoTokenizer
from common import read, get_model, load_setup, get_score, get_ents, base_name

out = sys.argv[1]
seed = sys.argv[2]

cfg, green = load_setup(out)
tok, model = get_model(base_name)


def tpr(pos, neg):
    neg = sorted(neg)
    k = math.ceil(0.99 * len(neg))
    tau = neg[k - 1]
    return sum(1 for s in pos if s > tau) / len(pos)


files = {"human": "human.jsonl", "human_short": "human_short.jsonl"}
for c in ["clean", "short", "substitution", "paraphrase"]:
    files[c] = "generated_" + c + "_" + seed + ".jsonl"

ids = {}
ents = {}
qs = {}
for c in files:
    rows = read(os.path.join(out, files[c]))
    ids[c] = [tok(r["text"], add_special_tokens=False)["input_ids"] for r in rows]
    ents[c], qs[c] = get_ents(model, ids[c], green=green)

print("gamma %.2f delta %.2f" % (cfg["gamma"], cfg["delta"]))
print("%-10s %8s %8s %8s %8s %8s" % ("score", "clean", "short", "subst", "para", "S_det"))
for kind in ["plain", "unique", "weighted", "mz", "oz"]:
    sc = {}
    for c in files:
        sc[c] = [get_score(ids[c][i], green, cfg["gamma"], kind, ents[c][i], qs[c][i], cfg["delta"]) for i in range(len(ids[c]))]
    row = []
    for c in ["clean", "short", "substitution", "paraphrase"]:
        if c == "short":
            row.append(tpr(sc[c], sc["human_short"]))
        else:
            row.append(tpr(sc[c], sc["human"]))
    print("%-10s %8.3f %8.3f %8.3f %8.3f %8.3f" % (kind, row[0], row[1], row[2], row[3], sum(row) / 4))
