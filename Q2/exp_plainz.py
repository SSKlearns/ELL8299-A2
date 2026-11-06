import os
import sys
import math
import numpy as np
from transformers import AutoTokenizer
from common import read, base_name, get_score

tok = AutoTokenizer.from_pretrained(base_name)
conds = ["clean", "short", "substitution", "paraphrase", "parasample"]
for tag in sys.argv[1:]:
    rows = []
    for s in [42, 43, 44, 45, 46]:
        d = "../out/q2_multi/" + tag + "/s" + str(s)
        green = np.load(d + "/green.npy")
        sc = {}
        for c in conds + ["human", "human_short"]:
            p = d + "/generated_" + c + "_" + str(s) + ".jsonl" if c in conds else d + "/" + c + ".jsonl"
            sc[c] = np.array([get_score(tok(r["text"], add_special_tokens=False)["input_ids"], green, 0.25, "plain") for r in read(p)])
        t = {}
        for c in conds:
            n = sc["human_short"] if c == "short" else sc["human"]
            tau = np.sort(n)[math.ceil(0.99 * len(n)) - 1]
            t[c] = float((sc[c] > tau).mean())
        rows.append([t["clean"], t["short"], t["substitution"], t["paraphrase"], t["parasample"],
                     (t["clean"] + t["short"] + t["substitution"] + t["paraphrase"]) / 4, (t["clean"] + t["short"] + t["substitution"] + t["parasample"]) / 4])
    o = np.mean(rows, 0)
    print("%-10s standard z-test, 5 seeds: clean %.3f short %.3f subst %.3f para %.3f parasample %.3f | S_det greedy %.3f sampled %.3f" % (tag, *o))
