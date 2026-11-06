import os
import sys
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from common import read, is_right, make_x, pick_abstain

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, ".."))
import eval as ev

out = sys.argv[1]
data = os.path.join(here, "..", "data", "public")
files = ["q3_qa_train", "q3_qa_dev", "q3_qa_dev_ood"]
rows = {}
feats = {}
for f in files:
    rows[f] = read(os.path.join(data, f + ".jsonl"))
    with open(os.path.join(out, "feats_" + f + ".pkl"), "rb") as fh:
        feats[f] = pickle.load(fh)
all_f = {}
for key in feats[files[0]]:
    if type(feats[files[0]][key]) is list:
        all_f[key] = sum([feats[f][key] for f in files], [])
    else:
        all_f[key] = np.concatenate([feats[f][key] for f in files])
src = np.concatenate([np.full(len(rows[f]), i) for i, f in enumerate(files)])
y = np.array([is_right(a, r["answers"]) for a, r in zip(all_f["answer"], sum([rows[f] for f in files], []))])
ood = src == 2
folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=0).split(np.zeros(len(y)), src * 2 + y))


def lr(c):
    return make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=3000))


def oof(x, c, w=None):
    p = np.zeros(len(y))
    for tr, te in folds:
        h = lr(c)
        if w is None:
            h.fit(x[tr], y[tr])
        else:
            h.fit(x[tr], y[tr], logisticregression__sample_weight=w[tr])
        p[te] = h.predict_proba(x[te])[:, 1]
    return p


pl = oof(all_f["hp14"], 0.05)
conf = oof(make_x(all_f, pl, "plain"), 1.0, np.where(ood, 5.0, 1.0))

print("=== accuracy by sample agreement (share of 8 samples matching greedy)")
ag = np.array(all_f["agree"])
for nm, m in [("id", ~ood), ("ood", ood)]:
    for lo, hi in [(0, 0.01), (0.01, 0.5), (0.5, 0.75), (0.75, 0.99), (0.99, 1.01)]:
        s = m & (ag >= lo) & (ag < hi)
        if s.sum() > 0:
            print("  %-4s agree in [%.2f, %.2f): n %4d acc %.3f mean conf %.3f" % (nm, lo, hi, s.sum(), y[s].mean(), conf[s].mean()))

print("\n=== top-n accuracy by OOF confidence (oracle view of selective)")
for nm, m in [("train", src == 0), ("dev", src == 1), ("ood", src == 2)]:
    c = conf[m]
    yy = y[m]
    order = np.argsort(-c)
    line = []
    for frac in [0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15]:
        n = max(1, int(round(frac * len(c))))
        line.append("%.2f:%.2f(c%.2f)" % (frac, yy[order[:n]].mean(), c[order[n - 1]]))
    best = 0
    for n in range(1, len(c) + 1):
        if 1 - yy[order[:n]].mean() <= 0.2:
            best = n
    print("  %-5s n %4d acc %.3f | %s | oracle max coverage at risk<=0.2: %.3f" % (nm, len(c), yy.mean(), " ".join(line), best / len(c)))

print("\n=== selective with a fixed confidence threshold chosen on OOF (pooled id = train+dev, ood separately)")
for t in [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]:
    line = []
    for nm, m in [("train", src == 0), ("dev", src == 1), ("ood", src == 2)]:
        s, cov, risk = ev.selective(list(y[m]), list(conf[m] < t))
        line.append("%s sel %.3f cov %.3f risk %.3f" % (nm, s, cov, risk))
    print("  t %.2f | %s" % (t, " | ".join(line)))

print("\n=== stacking several probes (OOF set score uses predicted-risk abstention, margin 0)")
keys = ["hp14", "ha14", "hp20", "ha20", "hp26", "ha26"]
probes = {}
for k in keys:
    probes[k] = oof(all_f[k], 0.05)


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def show(name, c):
    line = []
    for nm, m in [("train", src == 0), ("dev", src == 1), ("ood", src == 2)]:
        sc, a, e, s, cov, risk = ev.q3_score(list(c[m]), list(y[m]), pick_abstain(list(c[m]), 0.0))
        line.append("%s AUROC %.4f ECE %.4f set %.4f" % (nm, a, e, sc))
    print("  %-28s %s" % (name, " | ".join(line)))


from common import small_x
base_x = small_x(all_f)
w = np.where(ood, 5.0, 1.0)
show("no probe", oof(base_x, 1.0, w))
show("hp14 (current)", conf)
for combo in [["hp14", "ha14"], ["hp14", "hp20"], ["hp14", "ha14", "hp20", "ha20"], keys]:
    x = np.column_stack([base_x] + [logit(probes[k]) for k in combo])
    show("+".join(combo), oof(x, 1.0, w))
x = np.column_stack([base_x] + [logit(probes[k]) for k in ["hp14", "ha14"]])
x = np.column_stack([x, x * np.array(all_f["has_q"])[:, None]])
show("hp14+ha14 inter", oof(x, 1.0, w))
