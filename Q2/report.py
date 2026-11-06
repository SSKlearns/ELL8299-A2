import os
import sys
import math
import numpy as np
from common import read

folder = sys.argv[1]
conds = sys.argv[2:] if len(sys.argv) > 2 else ["clean", "short", "substitution", "paraphrase", "parasample"]


def load(c):
    p = os.path.join(folder, "scores_" + c + ".jsonl")
    if not os.path.exists(p):
        return None
    return np.array([r["score"] for r in read(p)], dtype=float)


neg = {"human": load("human"), "human_short": load("human_short")}
for k in neg:
    v = neg[k]
    print("%-12s n=%d mean %.3f sd %.3f max %.3f" % (k, len(v), v.mean(), v.std(), v.max()))
tpr_e = {}
tpr_g = {}
for c in conds:
    pos = load(c)
    if pos is None:
        continue
    n = neg["human_short"] if c == "short" else neg["human"]
    srt = np.sort(n)
    tau = srt[math.ceil(0.99 * len(n)) - 1]
    tau_g = n.mean() + 2.3263 * n.std()
    tpr_e[c] = float((pos > tau).mean())
    tpr_g[c] = float((pos > tau_g).mean())
    print("%-13s TPR emp %.3f (tau %.2f) | TPR gauss %.3f (tau %.2f) | pos mean %.2f median %.2f" % (c, tpr_e[c], tau, tpr_g[c], tau_g, pos.mean(), np.median(pos)))
for p in ["paraphrase", "parasample"]:
    if p in tpr_e:
        keys = ["clean", "short", "substitution", p]
        print("S_det with %-10s emp %.4f | gauss %.4f" % (p, sum(tpr_e[k] for k in keys) / 4, sum(tpr_g[k] for k in keys) / 4))
