import sys
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, SplineTransformer
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


for mode in ["fixed", "knn"]:
    f = sig[mode]
    other = sig["knn" if mode == "fixed" else "fixed"]
    y = np.array([is_right(a, r["answers"]) for a, r in zip(f["answer"], rows)])
    print("\n######## mode", mode, "| accuracy id %.4f dev %.4f ood %.4f" % (y[src < 2].mean(), y[src == 1].mean(), y[ood].mean()))
    feat = {}
    for key in ["lp_mean", "lp_min", "lp_first", "agree", "uniq", "has_q", "upper", "ptrue", "ptrue_b", "sem_ent"]:
        feat[key] = np.array(f[key], dtype=float)
    feat["n_tok"] = np.log1p(np.array(f["n_tok"], dtype=float))
    feat["ag_zero"] = np.array([1.0 if same(a, b) else 0.0 for a, b in zip(f["answer"], sig["zero"])])
    feat["ag_other"] = np.array([1.0 if same(a, b) else 0.0 for a, b in zip(f["answer"], other["answer"])])
    feat["knn_sim"] = np.array(sig["knn_sim"])
    for key in ["agree", "lp_mean", "ptrue", "ptrue_b", "ag_zero", "ag_other"]:
        print("   single %-8s AUROC id %.4f ood %.4f" % (key, roc_auc_score(y[src < 2], feat[key][src < 2]), roc_auc_score(y[ood], feat[key][ood])))

    def oof(x, c, w, seed, kind="lr"):
        folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(np.zeros(n), src * 2 + y))
        p = np.zeros(n)
        for tr, te in folds:
            if kind == "spline":
                h = make_pipeline(StandardScaler(), SplineTransformer(n_knots=4, degree=3), LogisticRegression(C=0.3, max_iter=3000))
            else:
                h = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=3000))
            if w is None:
                h.fit(x[tr], y[tr])
            else:
                h.fit(x[tr], y[tr], logisticregression__sample_weight=w[tr])
            p[te] = h.predict_proba(x[te])[:, 1]
        return p

    for key in ["hp14", "hp20", "ha14", "ha20"]:
        p = oof(f[key], 0.05, None, 0)
        print("   probe %-5s AUROC id %.4f ood %.4f" % (key, roc_auc_score(y[src < 2], p[src < 2]), roc_auc_score(y[ood], p[ood])))
    pls = [oof(f["hp14"], 0.05, None, sd) for sd in range(3)]
    w = np.where(ood, 5.0, 1.0)
    base = ["lp_mean", "lp_min", "lp_first", "n_tok", "agree", "uniq", "has_q", "upper"]

    def evaluate(name, extra, kind="lr"):
        out = []
        for sd in range(3):
            x = np.column_stack([feat[k2] for k2 in base + extra] + [logit(pls[sd])])
            p = oof(x, 1.0, w, sd, kind)
            r = []
            for m in [src < 2, src == 1, ood]:
                sc, a, e, s, cov, risk = q3_score(list(p[m]), list(y[m]), [True] * m.sum())
                s2 = selective(list(y[m]), exp_pick(p[m]))[0]
                r += [a, e, (a + 1 - e) / 3, s2]
            out.append(r)
        o = np.mean(out, 0)
        print("   %-34s id AUROC %.4f ECE %.4f | dev AUROC %.4f ECE %.4f | ood AUROC %.4f ECE %.4f | set w/o sel id %.4f ood %.4f mean %.4f | EV-rule sel id %.4f dev %.4f ood %.4f" % (
            name, o[0], o[1], o[4], o[5], o[8], o[9], o[2], o[10], (o[2] + o[10]) / 2, o[3], o[7], o[11]))

    evaluate("v1 features", [])
    evaluate("+ ptrue", ["ptrue"])
    evaluate("+ ptrue_b", ["ptrue_b"])
    evaluate("+ ptrue + ptrue_b", ["ptrue", "ptrue_b"])
    evaluate("+ sem_ent", ["sem_ent"])
    evaluate("+ ag_zero + ag_other", ["ag_zero", "ag_other"])
    evaluate("+ knn_sim", ["knn_sim"])
    evaluate("+ ptrue + ptrue_b + ag_zero + ag_other", ["ptrue", "ptrue_b", "ag_zero", "ag_other"])
    evaluate("+ all", ["ptrue", "ptrue_b", "sem_ent", "ag_zero", "ag_other", "knn_sim"])
    evaluate("+ all, spline calibrator", ["ptrue", "ptrue_b", "sem_ent", "ag_zero", "ag_other", "knn_sim"], "spline")
