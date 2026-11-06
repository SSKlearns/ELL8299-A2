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
print("single ag_ks AUROC id %.4f ood %.4f" % (roc_auc_score(y[src < 2], feat["ag_ks"][src < 2]), roc_auc_score(y[ood], feat["ag_ks"][ood])))
base = ["lp_mean", "lp_min", "lp_first", "n_tok", "agree", "uniq", "has_q", "upper"]
seeds = [0, 1, 2, 3, 4]


def oof(x, c, w, seed):
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(np.zeros(n), src * 2 + y))
    p = np.zeros(n)
    for tr, te in folds:
        h = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=3000))
        if w is None:
            h.fit(x[tr], y[tr])
        else:
            h.fit(x[tr], y[tr], logisticregression__sample_weight=w[tr])
        p[te] = h.predict_proba(x[te])[:, 1]
    return p


pls = [oof(f["hp14"], 0.05, None, sd) for sd in seeds]
store = {}


def evaluate(name, extra, c, wo):
    out = []
    ps = []
    w = np.where(ood, wo, 1.0)
    for i, sd in enumerate(seeds):
        x = np.column_stack([feat[k] for k in base + extra] + [logit(pls[i])])
        p = oof(x, c, w, sd)
        ps.append(p)
        r = []
        for m in [src < 2, ood]:
            sc, a, e, s, cov, risk = q3_score(list(p[m]), list(y[m]), [True] * m.sum())
            r += [a, e, (a + 1 - e) / 3]
        out.append(r)
    o = np.mean(out, 0)
    sd_ = np.std([(r[2] + r[5]) / 2 for r in out])
    store[name] = ps
    print("%-48s id AUROC %.4f ECE %.4f | ood AUROC %.4f ECE %.4f | set w/o sel id %.4f ood %.4f mean %.4f (sd over fold seeds %.4f)" % (
        name, o[0], o[1], o[3], o[4], o[2], o[5], (o[2] + o[5]) / 2, sd_))


new = ["ptrue", "ptrue_b", "ag_zero", "ag_other"]
evaluate("v1 C=1 w=5", [], 1.0, 5.0)
for c in [0.1, 0.3, 1.0]:
    for wo in [1.0, 3.0, 5.0]:
        evaluate("new C=%g w=%g" % (c, wo), new, c, wo)
evaluate("new + ag_ks C=1 w=5", new + ["ag_ks"], 1.0, 5.0)
evaluate("new + ag_ks C=0.3 w=3", new + ["ag_ks"], 0.3, 3.0)

print("\npaired bootstrap over questions, new C=1 w=5 minus v1, fold seed 0")
a = store["v1 C=1 w=5"][0]
b = store["new C=1 w=5"][0]
rng = np.random.default_rng(0)
for nm, m in [("id", src < 2), ("dev", src == 1), ("ood", ood)]:
    ix = np.where(m)[0]
    d = []
    for t in range(1000):
        s = rng.choice(ix, len(ix))
        if y[s].sum() == 0:
            continue
        d.append(roc_auc_score(y[s], b[s]) - roc_auc_score(y[s], a[s]))
    d = np.array(d)
    print("   %-4s AUROC gain %.4f  95%% CI [%.4f, %.4f]  P(gain <= 0) %.3f" % (nm, d.mean(), np.percentile(d, 2.5), np.percentile(d, 97.5), (d <= 0).mean()))
