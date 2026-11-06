import sys
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
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
seeds = [0, 1, 2, 3, 4]
strat = src * 2


def feats(me, other):
    y = np.array([is_right(a, r["answers"]) for a, r in zip(me["answer"], rows)])
    x = {}
    for key in ["lp_mean", "lp_min", "lp_first", "agree", "uniq", "has_q", "upper", "ptrue", "ptrue_b"]:
        x[key] = np.array(me[key], dtype=float)
    x["n_tok"] = np.log1p(np.array(me["n_tok"], dtype=float))
    x["ag_zero"] = np.array([1.0 if same(a, b) else 0.0 for a, b in zip(me["answer"], sig["zero"])])
    x["ag_other"] = np.array([1.0 if same(a, b) else 0.0 for a, b in zip(me["answer"], other["answer"])])
    x["ag_os"] = np.array([sum(1 for s in other["samples"][i] if same(s, me["answer"][i])) / 8 for i in range(n)])
    return y, np.column_stack([x[k] for k in sorted(x)])


def lr(c):
    return make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=3000))


def two_stage(x, y, tr, te, seed, c=0.3):
    inner = np.zeros(len(tr))
    sub = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=seed + 100).split(np.zeros(len(tr)), strat[tr]))
    for a, b in sub:
        h = lr(c)
        h.fit(x[tr][a], y[tr][a])
        inner[b] = h.decision_function(x[tr][b])
    h = lr(c)
    h.fit(x[tr], y[tr])
    z = h.decision_function(x[te])
    p = np.zeros(len(te))
    pin = np.zeros(len(tr))
    for dom in [False, True]:
        mtr = ood[tr] == dom
        mte = ood[te] == dom
        cal = LogisticRegression(C=100.0, max_iter=3000)
        cal.fit(inner[mtr][:, None], y[tr][mtr])
        p[mte] = cal.predict_proba(z[mte][:, None])[:, 1]
        pin[mtr] = cal.predict_proba(inner[mtr][:, None])[:, 1]
    return p, pin


def probe(key, me, y, tr, te):
    h = lr(0.05)
    h.fit(me[key][tr], y[tr])
    return h.predict_proba(me[key][te])[:, 1]


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


yf, xf = feats(sig["fixed"], sig["knn"])
yk, xk = feats(sig["knn"], sig["fixed"])
res = {"fixed": [], "choose": []}
for sd in seeds:
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=sd).split(np.zeros(n), strat))
    pf = np.zeros(n)
    pk = np.zeros(n)
    for tr, te in folds:
        inner_pf = np.zeros(len(tr))
        out = []
        for me, y, x in [(sig["fixed"], yf, xf), (sig["knn"], yk, xk)]:
            pl = np.zeros(n)
            sub = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=sd + 7).split(np.zeros(len(tr)), strat[tr]))
            for a, b in sub:
                pl[tr[b]] = probe("hp14", me, y, tr[a], tr[b])
            pl[te] = probe("hp14", me, y, tr, te)
            xx = np.column_stack([x, logit(pl)])
            p, pin = two_stage(xx, y, tr, te, sd)
            out.append(p)
        pf[te] = out[0]
        pk[te] = out[1]
    pick_k = np.array([(not same(a, b)) and pk[i] > pf[i] for i, (a, b) in enumerate(zip(sig["fixed"]["answer"], sig["knn"]["answer"]))])
    yc = np.where(pick_k, yk, yf)
    zc = logit(np.where(pick_k, pk, pf))
    pc = np.zeros(n)
    folds2 = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=sd + 50).split(np.zeros(n), strat * 2 + yc))
    for tr, te in folds2:
        for dom in [False, True]:
            mtr = tr[ood[tr] == dom]
            mte = te[ood[te] == dom]
            cal = LogisticRegression(C=100.0, max_iter=3000)
            cal.fit(np.column_stack([zc[mtr], pick_k[mtr]]), yc[mtr])
            pc[mte] = cal.predict_proba(np.column_stack([zc[mte], pick_k[mte]]))[:, 1]
    for name, p, y in [("fixed", pf, yf), ("choose", pc, yc)]:
        r = []
        for m in [src < 2, src == 1, ood]:
            sc, a, e, s, cov, risk = q3_score(list(p[m]), list(y[m]), [True] * m.sum())
            r += [y[m].mean(), a, e, (a + 1 - e) / 3]
        for m in [src == 0, src == 1, ood]:
            r.append(selective(list(y[m]), exp_pick(p[m]))[0])
        res[name].append(r)
    print("seed", sd, "picked retrieved answer on %d questions" % pick_k.sum())
for name in res:
    o = np.mean(res[name], 0)
    print("%-7s acc id %.4f dev %.4f ood %.4f | AUROC id %.4f dev %.4f ood %.4f | ECE id %.4f ood %.4f | set w/o sel id %.4f ood %.4f mean %.4f | EV sel train %.4f dev %.4f ood %.4f" % (
        name, o[0], o[4], o[8], o[1], o[5], o[9], o[2], o[10], o[3], o[11], (o[3] + o[11]) / 2, o[12], o[13], o[14]))
