import sys
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from common import read, is_right, same, q3_score, selective, logit

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
with open(sys.argv[1] + "/sig2.pkl", "rb") as fh:
    sig = pickle.load(fh)
with open(sys.argv[1] + "/sig3.pkl", "rb") as fh:
    s3 = pickle.load(fh)
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
feat["ptrue_k"] = np.array(s3["ptrue_k"])
feat["ptrue_f"] = np.array(s3["ptrue_f"])
print("ptrue_b recomputed equals sig2 value: max abs diff %.4f" % np.max(np.abs(np.array(s3["ptrue_b"]) - feat["ptrue_b"])))
for key in ["ptrue_b", "ptrue_k", "ptrue_f"]:
    print("single %-8s AUROC id %.4f dev %.4f ood %.4f" % (key, roc_auc_score(y[src < 2], feat[key][src < 2]), roc_auc_score(y[src == 1], feat[key][src == 1]), roc_auc_score(y[ood], feat[key][ood])))
base = ["lp_mean", "lp_min", "lp_first", "n_tok", "agree", "uniq", "has_q", "upper"]
new = ["ptrue", "ptrue_b", "ag_zero", "ag_other", "ag_ks"]
seeds = [0, 1, 2, 3, 4]


def lr(c):
    return make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=3000))


def folds_of(seed):
    return list(StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(np.zeros(n), src * 2 + y))


def oof(x, c, seed):
    p = np.zeros(n)
    for tr, te in folds_of(seed):
        h = lr(c)
        h.fit(x[tr], y[tr])
        p[te] = h.predict_proba(x[te])[:, 1]
    return p


def two_stage(x, c, seed):
    p = np.zeros(n)
    for tr, te in folds_of(seed):
        inner = np.zeros(len(tr))
        sub = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=seed + 100).split(np.zeros(len(tr)), src[tr] * 2 + y[tr]))
        for a, b in sub:
            h = lr(c)
            h.fit(x[tr][a], y[tr][a])
            inner[b] = h.decision_function(x[tr][b])
        h = lr(c)
        h.fit(x[tr], y[tr])
        z = h.decision_function(x[te])
        for dom in [False, True]:
            mtr = ood[tr] == dom
            mte = ood[te] == dom
            cal = LogisticRegression(C=100.0, max_iter=3000)
            cal.fit(inner[mtr][:, None], y[tr][mtr])
            p[te[mte]] = cal.predict_proba(z[mte][:, None])[:, 1]
    return p


def exp_pick(conf, sims=4000):
    rng = np.random.default_rng(0)
    conf = np.asarray(conf)
    order = np.argsort(-conf)
    top = min(len(conf), 200)
    draws = rng.random((sims, top)) < conf[order[:top]]
    nn = np.arange(1, top + 1)
    risk = np.cumsum(~draws, 1) / nn
    ev = (nn / len(conf) * np.clip(1 - (risk - 0.2) / 0.1, 0, 1)).mean(0)
    best = int(np.argmax(ev)) + 1
    keep = set(order[:best].tolist())
    return [i not in keep for i in range(len(conf))]


probes = {}
for key in ["hp14", "hv14", "hv20", "hv26"]:
    src_f = f if key.startswith("hp") else s3
    probes[key] = {sd: oof(src_f[key], 0.05, sd) for sd in seeds}
    p = probes[key][0]
    print("probe %-5s AUROC id %.4f dev %.4f ood %.4f" % (key, roc_auc_score(y[src < 2], p[src < 2]), roc_auc_score(y[src == 1], p[src == 1]), roc_auc_score(y[ood], p[ood])))


def show(name, extra, pk, c=0.3):
    out = []
    for sd in seeds:
        x = np.column_stack([feat[k] for k in base + extra] + [logit(probes[k][sd]) for k in pk])
        p = two_stage(x, c, sd)
        r = []
        for m in [src < 2, ood]:
            sc, a, e, s, cov, risk = q3_score(list(p[m]), list(y[m]), [True] * m.sum())
            r += [a, e, (a + 1 - e) / 3]
        for m in [src == 0, src == 1, ood]:
            r.append(selective(list(y[m]), exp_pick(p[m]))[0])
        out.append(r)
    o = np.mean(out, 0)
    print("%-40s id AUROC %.4f ECE %.4f | ood AUROC %.4f ECE %.4f | set w/o sel id %.4f ood %.4f mean %.4f | EV-rule sel train %.4f dev %.4f ood %.4f" % (
        name, o[0], o[1], o[3], o[4], o[2], o[5], (o[2] + o[5]) / 2, o[6], o[7], o[8]))


show("v1 features (two stage)", [], ["hp14"])
show("A = new (two stage)", new, ["hp14"])
show("A + ptrue_k", new + ["ptrue_k"], ["hp14"])
show("A + ptrue_f", new + ["ptrue_f"], ["hp14"])
show("A + hv14 probe", new, ["hp14", "hv14"])
show("A + hv20 probe", new, ["hp14", "hv20"])
show("A + ptrue_k + ptrue_f + hv20", new + ["ptrue_k", "ptrue_f"], ["hp14", "hv20"])
