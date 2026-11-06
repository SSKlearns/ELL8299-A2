import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, SplineTransformer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from common import read, is_right, make_x, q3_score, pick_abstain, small_x, logit

files = ["q3_qa_train", "q3_qa_dev", "q3_qa_dev_ood"]
rows = []
src = []
feats = {}
for i, f in enumerate(files):
    r = read("../data/public/" + f + ".jsonl")
    rows += r
    src += [i] * len(r)
    with open("../out/q3/feats_" + f + ".pkl", "rb") as fh:
        feats[f] = pickle.load(fh)
src = np.array(src)
all_f = {}
for key in feats[files[0]]:
    if type(feats[files[0]][key]) is list:
        all_f[key] = sum([feats[f][key] for f in files], [])
    else:
        all_f[key] = np.concatenate([feats[f][key] for f in files])
y = np.array([is_right(a, r["answers"]) for a, r in zip(all_f["answer"], rows)])
ood = src == 2
w = np.where(ood, 5.0, 1.0)


def run(make, x, seeds=[0, 1, 2]):
    res = []
    for sd in seeds:
        folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=sd).split(np.zeros(len(y)), src * 2 + y))
        p = np.zeros(len(y))
        for tr, te in folds:
            h = make()
            h.fit(x[tr], y[tr], **({} if w is None else {make.wkey: w[tr]}))
            p[te] = h.predict_proba(x[te])[:, 1]
        res.append(p)
    return res


def lr(c):
    def f():
        return make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=3000))
    f.wkey = "logisticregression__sample_weight"
    return f


def score(ps, name):
    out = []
    for p in ps:
        r = []
        for m in [src < 2, ood]:
            sc, a, e, s, cov, risk = q3_score(list(p[m]), list(y[m]), pick_abstain(list(p[m]), 0.0))
            r += [a, e, sc]
        out.append(r)
    o = np.mean(out, 0)
    print("%-42s id AUROC %.4f ECE %.4f set %.4f | ood AUROC %.4f ECE %.4f set %.4f | mean %.4f" % (name, o[0], o[1], o[2], o[3], o[4], o[5], (o[2] + o[5]) / 2))


def probe(key, c, seed):
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(np.zeros(len(y)), src * 2 + y))
    p = np.zeros(len(y))
    x = all_f[key]
    for tr, te in folds:
        h = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=3000))
        h.fit(x[tr], y[tr])
        p[te] = h.predict_proba(x[te])[:, 1]
    return p


pl = probe("hp14", 0.05, 0)
x = make_x(all_f, pl, "plain")
score(run(lr(1.0), x), "current: LR C=1 (probe OOF seed 0)")
score(run(lr(0.1), x), "LR C=0.1")
score(run(lr(10.0), x), "LR C=10")


def spl():
    return make_pipeline(StandardScaler(), SplineTransformer(n_knots=4, degree=3), LogisticRegression(C=0.3, max_iter=3000))
spl.wkey = "logisticregression__sample_weight"
score(run(spl, x), "spline-LR (additive non-linear)")


def hgb():
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=40, l2_regularization=1.0)
hgb.wkey = "sample_weight"
score(run(hgb, x), "gradient boosting depth 3")

for c in [0.01, 0.02, 0.1]:
    p2 = probe("hp14", c, 0)
    score(run(lr(1.0), make_x(all_f, p2, "plain")), "probe C=%g" % c)
pc = np.column_stack([all_f["hp14"], all_f["hp20"]])
all_f["hp1420"] = pc
p3 = probe("hp1420", 0.02, 0)
score(run(lr(1.0), make_x(all_f, p3, "plain")), "probe on layers 14+20 concatenated")
