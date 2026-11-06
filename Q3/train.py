import os
import sys
import json
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from common import read, write, get_model, get_feats, is_right, pick_shots, make_x, make_x_ood, pick_abstain, layers, save_lr, q3_score, get_dom, platt

out = sys.argv[1]
data = sys.argv[2]
os.makedirs(out, exist_ok=True)
k = 8
if len(sys.argv) > 3:
    k = int(sys.argv[3])
c_cal = 0.3
files = ["q3_qa_train", "q3_qa_dev", "q3_qa_dev_ood"]
rows = {}
for f in files:
    rows[f] = read(os.path.join(data, f + ".jsonl"))
shots = pick_shots(rows["q3_qa_train"])
print("shots:", shots)
pool = []
for f in files:
    for r in rows[f]:
        pool.append([r["question"], r["answers"][0]])
write(os.path.join(out, "pool.jsonl"), [{"question": a, "answer": b} for a, b in pool])

tok = None
model = None
feats = {}
for f in files:
    path = os.path.join(out, "feats2_" + f + ".pkl")
    if os.path.exists(path):
        with open(path, "rb") as fh:
            feats[f] = pickle.load(fh)
        continue
    if model is None:
        tok, model = get_model()
    feats[f] = get_feats(tok, model, rows[f], shots, pool, k=k)
    with open(path, "wb") as fh:
        pickle.dump(feats[f], fh)

all_f = {}
for key in feats[files[0]]:
    if type(feats[files[0]][key]) is list:
        all_f[key] = sum([feats[f][key] for f in files], [])
    else:
        all_f[key] = np.concatenate([feats[f][key] for f in files])
src = np.concatenate([np.full(len(rows[f]), i) for i, f in enumerate(files)])
y = np.array([is_right(a, r["answers"]) for a, r in zip(all_f["answer"], sum([rows[f] for f in files], []))])
ood = src == 2
dom = get_dom(all_f)
print("domain rule agrees with file of origin on %d / %d questions" % ((dom == ood).sum(), len(dom)))
for i, f in enumerate(files):
    print("%-15s accuracy of greedy answer %.4f" % (f, y[src == i].mean()))

folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=0).split(np.zeros(len(y)), src * 2 + y))


def lr(c):
    return make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=3000))


def oof(x, c):
    p = np.zeros(len(y))
    for tr, te in folds:
        h = lr(c)
        h.fit(x[tr], y[tr])
        p[te] = h.predict_proba(x[te])[:, 1]
    return p


def fit_two(x, yy, d, strat):
    inner = np.zeros(len(yy))
    for a, b in StratifiedKFold(n_splits=5, shuffle=True, random_state=1).split(np.zeros(len(yy)), strat):
        h = lr(c_cal)
        h.fit(x[a], yy[a])
        inner[b] = h.decision_function(x[b])
    h = lr(c_cal)
    h.fit(x, yy)
    maps = {}
    for v in [0, 1]:
        m = d == v
        cal = LogisticRegression(C=100.0, max_iter=3000)
        cal.fit(inner[m][:, None], yy[m])
        maps[str(v)] = [float(cal.coef_[0][0]), float(cal.intercept_[0])]
    return h, maps


def report(name, c):
    res = []
    for nm, m in [("dev(id)", src == 1), ("dev_ood", src == 2), ("train", src == 0)]:
        sc, a, e, s, cov, risk = q3_score(list(c[m]), list(y[m]), pick_abstain(list(c[m])))
        res.append(sc)
        print("  %-24s %-8s AUROC %.4f ECE %.4f sel %.4f (cov %.3f risk %.3f) set %.4f" % (name, nm, a, e, s, cov, risk, sc))
    return res


print("\n=== baseline: confidence = exp(mean token logprob), abstain if conf < 0.5")
base = np.exp(np.array(all_f["lp_mean"]))
for nm, m in [("dev(id)", src == 1), ("dev_ood", src == 2)]:
    sc, a, e, s, cov, risk = q3_score(list(base[m]), list(y[m]), list(base[m] < 0.5))
    print("  baseline %-8s AUROC %.4f ECE %.4f sel %.4f (cov %.3f risk %.3f) set %.4f" % (nm, a, e, s, cov, risk, sc))

print("\n=== hidden state probes (OOF AUROC)")
best = None
for l in layers:
    for pos in ["hp", "ha"]:
        key = pos + str(l)
        p = oof(all_f[key], 0.05)
        a_id = roc_auc_score(y[~ood], p[~ood])
        a_od = roc_auc_score(y[ood], p[ood])
        print("  %-5s auroc id %.4f ood %.4f mean %.4f" % (key, a_id, a_od, (a_id + a_od) / 2))
        if best is None or (a_id + a_od) / 2 > best[0]:
            best = ((a_id + a_od) / 2, key, p)
key = best[1]
pl = best[2]
print("picked probe", key)

print("\n=== calibrators, 5-fold out-of-fold: in-domain questions use the two-stage model on all features, out-of-domain questions use the small model")
x = make_x(all_f, pl)
x_ood = make_x_ood(all_f, pl)
w = np.where(dom == 1, 5.0, 1.0)
strat = src * 2 + y
c_two = np.zeros(len(y))
c_small = np.zeros(len(y))
for tr, te in folds:
    h, maps = fit_two(x[tr], y[tr], dom[tr], strat[tr])
    c_two[te] = platt(h.decision_function(x[te]), dom[te], maps)
    h = lr(1.0)
    h.fit(x_ood[tr], y[tr], logisticregression__sample_weight=w[tr])
    c_small[te] = h.predict_proba(x_ood[te])[:, 1]
report("two-stage, all features", c_two)
report("small model", c_small)
c = np.where(dom == 1, c_small, c_two)
ss = report("routed by domain", c)
print("OOF mean set score of dev(id) and dev_ood: %.4f" % ((ss[0] + ss[1]) / 2))

probe = lr(0.05)
probe.fit(all_f[key], y)
h, maps = fit_two(x, y, dom, strat)
h_ood = lr(1.0)
h_ood.fit(x_ood, y, logisticregression__sample_weight=w)
save_lr(os.path.join(out, "probe.npz"), probe)
save_lr(os.path.join(out, "calib.npz"), h)
save_lr(os.path.join(out, "calib_ood.npz"), h_ood)
cfg = {"shots": shots, "k": k, "key": key, "c_cal": c_cal, "maps": maps, "oof_mean_set": float((ss[0] + ss[1]) / 2)}
with open(os.path.join(out, "config.json"), "w") as fh:
    json.dump(cfg, fh, indent=1)
print("saved to", out, cfg)
