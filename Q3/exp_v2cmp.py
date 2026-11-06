import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from common import read, is_right, make_x, q3_score, pick_abstain, get_dom, logit, selective

files = ["q3_qa_train", "q3_qa_dev", "q3_qa_dev_ood"]
rows = []
src = []
feats = {}
for i, f in enumerate(files):
    r = read("../data/public/" + f + ".jsonl")
    rows += r
    src += [i] * len(r)
    with open("../out/q3_v2/feats2_" + f + ".pkl", "rb") as fh:
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
dom = get_dom(all_f)
n = len(y)
strat = src * 2 + y


def lr(c):
    return make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=3000))


def v1_x(pl):
    return np.column_stack([np.array(all_f[k], dtype=float) for k in ["lp_mean", "lp_min", "lp_first"]] + [np.log1p(all_f["n_tok"])] +
                           [np.array(all_f[k], dtype=float) for k in ["agree", "uniq", "has_q", "upper"]] + [logit(pl)])


def old_rule(conf):
    order = sorted(range(len(conf)), key=lambda i: -conf[i])
    best = 0
    total = 0.0
    for m in range(1, len(order) + 1):
        total += conf[order[m - 1]]
        if total / m >= 0.8:
            best = m
    keep = set(order[:best])
    return [i not in keep for i in range(len(conf))]


def two(x, tr, te, sd, kind):
    inner = np.zeros(len(tr))
    for a, b in StratifiedKFold(n_splits=5, shuffle=True, random_state=sd + 1).split(np.zeros(len(tr)), strat[tr]):
        h = lr(0.3)
        h.fit(x[tr][a], y[tr][a])
        inner[b] = h.decision_function(x[tr][b])
    h = lr(0.3)
    h.fit(x[tr], y[tr])
    z = h.decision_function(x[te])
    if kind == "sep":
        p = np.zeros(len(te))
        for v in [0, 1]:
            cal = LogisticRegression(C=100.0, max_iter=3000)
            cal.fit(inner[dom[tr] == v][:, None], y[tr][dom[tr] == v])
            p[dom[te] == v] = cal.predict_proba(z[dom[te] == v][:, None])[:, 1]
        return p
    cal = LogisticRegression(C=100.0, max_iter=3000)
    cal.fit(np.column_stack([inner, dom[tr]]), y[tr])
    return cal.predict_proba(np.column_stack([z, dom[te]]))[:, 1]


res = {}
for sd in range(10):
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=sd).split(np.zeros(n), strat))
    pl = np.zeros(n)
    for tr, te in folds:
        h = lr(0.05)
        h.fit(all_f["hp14"][tr], y[tr])
        pl[te] = h.predict_proba(all_f["hp14"][te])[:, 1]
    x1 = v1_x(pl)
    x2 = make_x(all_f, pl)
    outs = {"v1 (one stage, w_ood 5, old 0.8 rule)": np.zeros(n), "v1 + EV rule": None, "v2 one stage (w_ood 3, C 0.3)": np.zeros(n),
            "v2 two stage, per-domain slope+offset": np.zeros(n), "v2 two stage, shared slope, per-domain offset": np.zeros(n)}
    for tr, te in folds:
        h = lr(1.0)
        h.fit(x1[tr], y[tr], logisticregression__sample_weight=np.where(ood[tr], 5.0, 1.0))
        outs["v1 (one stage, w_ood 5, old 0.8 rule)"][te] = h.predict_proba(x1[te])[:, 1]
        h = lr(0.3)
        h.fit(x2[tr], y[tr], logisticregression__sample_weight=np.where(ood[tr], 3.0, 1.0))
        outs["v2 one stage (w_ood 3, C 0.3)"][te] = h.predict_proba(x2[te])[:, 1]
        outs["v2 two stage, per-domain slope+offset"][te] = two(x2, tr, te, sd, "sep")
        outs["v2 two stage, shared slope, per-domain offset"][te] = two(x2, tr, te, sd, "shared")
    outs["v1 + EV rule"] = outs["v1 (one stage, w_ood 5, old 0.8 rule)"]
    for name, p in outs.items():
        r = []
        for m in [src == 1, ood, src == 0]:
            ab = old_rule(list(p[m])) if name.startswith("v1 (") else pick_abstain(list(p[m]))
            sc, a, e, s, cov, risk = q3_score(list(p[m]), list(y[m]), ab)
            r += [a, e, s, sc]
        res.setdefault(name, []).append(r)
print("10 fold seeds; mean (sd).  dev = 500 ID, ood = 200 OOD, train = 2000 ID")
for name, v in res.items():
    o = np.mean(v, 0)
    sdv = np.std(v, 0)
    mean_set = [(r[3] + r[7]) / 2 for r in v]
    print("%-48s dev AUROC %.4f ECE %.4f sel %.4f set %.4f | ood AUROC %.4f ECE %.4f sel %.4f set %.4f | train set %.4f sel %.4f | MEAN(dev,ood) %.4f (sd %.4f)" % (
        name, o[0], o[1], o[2], o[3], o[4], o[5], o[6], o[7], o[11], o[10], np.mean(mean_set), np.std(mean_set)))
