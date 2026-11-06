import sys
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from common import read, is_right, same, q3_score, ece, logit

files = ["q3_qa_train", "q3_qa_dev", "q3_qa_dev_ood"]
rows = []
src = []
for i, f in enumerate(files):
    r = read("../data/public/" + f + ".jsonl")
    rows += r
    src += [i] * len(r)
src = np.array(src)
ood = src == 2
n = len(rows)
with open(sys.argv[1], "rb") as fh:
    sig = pickle.load(fh)
f = sig["fixed"]
g = sig["knn"]
y = np.array([is_right(a, r["answers"]) for a, r in zip(f["answer"], rows)])
feat = {}
for key in ["lp_mean", "lp_min", "lp_first", "agree", "uniq", "has_q", "upper", "ptrue", "ptrue_b"]:
    feat[key] = np.array(f[key], dtype=float)
feat["n_tok"] = np.log1p(np.array(f["n_tok"], dtype=float))
feat["ag_zero"] = np.array([1.0 if same(a, b) else 0.0 for a, b in zip(f["answer"], sig["zero"])])
feat["ag_other"] = np.array([1.0 if same(a, b) else 0.0 for a, b in zip(f["answer"], g["answer"])])
feat["ag_ks"] = np.array([sum(1 for s in g["samples"][i] if same(s, f["answer"][i])) / 8 for i in range(n)])
base = ["lp_mean", "lp_min", "lp_first", "n_tok", "agree", "uniq", "has_q", "upper"]
new = ["ptrue", "ptrue_b", "ag_zero", "ag_other", "ag_ks"]
seeds = [0, 1, 2, 3, 4]


def lr(c):
    return make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=3000))


def folds_of(seed):
    return list(StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(np.zeros(n), src * 2 + y))


def oof(x, c, w, seed):
    p = np.zeros(n)
    for tr, te in folds_of(seed):
        h = lr(c)
        h.fit(x[tr], y[tr], logisticregression__sample_weight=w[tr])
        p[te] = h.predict_proba(x[te])[:, 1]
    return p


def two_stage(x, c, w, seed):
    p = np.zeros(n)
    for tr, te in folds_of(seed):
        inner = np.zeros(len(tr))
        sub = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=seed + 100).split(np.zeros(len(tr)), src[tr] * 2 + y[tr]))
        for a, b in sub:
            h = lr(c)
            h.fit(x[tr][a], y[tr][a], logisticregression__sample_weight=w[tr][a])
            inner[b] = h.decision_function(x[tr][b])
        h = lr(c)
        h.fit(x[tr], y[tr], logisticregression__sample_weight=w[tr])
        z = h.decision_function(x[te])
        for dom in [0, 1]:
            mtr = ood[tr] == bool(dom)
            mte = ood[te] == bool(dom)
            cal = LogisticRegression(C=100.0, max_iter=3000)
            cal.fit(inner[mtr][:, None], y[tr][mtr])
            p[te[mte]] = cal.predict_proba(z[mte][:, None])[:, 1]
    return p


def floor(p, sims=2000):
    rng = np.random.default_rng(0)
    return np.mean([ece(list(p), list((rng.random(len(p)) < p).astype(int))) for t in range(sims)])


def show(name, fn):
    out = []
    fl = []
    for sd in seeds:
        p = fn(sd)
        r = []
        for m in [src < 2, ood]:
            sc, a, e, s, cov, risk = q3_score(list(p[m]), list(y[m]), [True] * m.sum())
            r += [a, e, (a + 1 - e) / 3]
        out.append(r)
        if sd == 0:
            fl = [floor(p[src == 1]), floor(p[ood]), floor(np.resize(p[ood], 1000))]
    o = np.mean(out, 0)
    print("%-44s id AUROC %.4f ECE %.4f | ood AUROC %.4f ECE %.4f | set w/o sel id %.4f ood %.4f mean %.4f | ECE floor if perfectly calibrated: dev(500) %.4f ood(200) %.4f ood(1000) %.4f" % (
        name, o[0], o[1], o[3], o[4], o[2], o[5], (o[2] + o[5]) / 2, fl[0], fl[1], fl[2]))


pls = {sd: oof(f["hp14"], 0.05, np.ones(n), sd) for sd in seeds}


def xs(extra, sd):
    return np.column_stack([feat[k] for k in base + extra] + [logit(pls[sd])])


show("v1, one stage C=1 w=5", lambda sd: oof(xs([], sd), 1.0, np.where(ood, 5.0, 1.0), sd))
show("new, one stage C=0.3 w=3", lambda sd: oof(xs(new, sd), 0.3, np.where(ood, 3.0, 1.0), sd))
show("v1, two stage (per-domain Platt) w=1", lambda sd: two_stage(xs([], sd), 1.0, np.ones(n), sd))
show("new, two stage (per-domain Platt) w=1", lambda sd: two_stage(xs(new, sd), 0.3, np.ones(n), sd))
show("new, two stage (per-domain Platt) w=3", lambda sd: two_stage(xs(new, sd), 0.3, np.where(ood, 3.0, 1.0), sd))
