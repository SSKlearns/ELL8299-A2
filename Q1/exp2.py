import os
import pickle
import numpy as np
from common import read, bbq_score

here = os.path.dirname(os.path.abspath(__file__))
data = os.path.join(here, "..", "data", "public")
cache = os.path.join(here, "..", "out", "q1_exp")
train = read(os.path.join(data, "q1_bbq_train.jsonl"))
dev = read(os.path.join(data, "q1_bbq_dev.jsonl"))
with open(os.path.join(cache, "exp_res.pkl"), "rb") as f:
    z = pickle.load(f)
res = z["res"]
u_tr = z["u_tr"]
u_dv = z["u_dv"]
pk_tr, pk_dv = z["pickers"]["letter"]
thrs = [round(0.01 * i, 2) for i in range(1, 100)]


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def otsu(v):
    v = np.sort(v)
    best = None
    for i in range(1, len(v)):
        a = v[:i]
        b = v[i:]
        s = len(a) * len(b) * (a.mean() - b.mean()) ** 2
        if best is None or s > best[0]:
            best = (s, (v[i - 1] + v[i]) / 2)
    return best[1]


def route(p, t, pk, unk):
    return [unk[i] if p[i] >= t[i] else pk[i] for i in range(len(p))]


def by_cat(rows, p, fn):
    t = np.zeros(len(rows))
    cats = np.array([r["category"] for r in rows])
    for c in set(cats):
        m = cats == c
        t[m] = fn(p[m])
    return t


def tol_thr(p):
    ss = [bbq_score(route(p, [t] * len(p), pk_tr, u_tr), train) for t in thrs]
    top = max(ss)
    for i in range(len(thrs)):
        if ss[i] >= top - 0.01:
            return thrs[i]


ens = {}
for a in [False, True]:
    ens[("ens_last_all", a)] = (np.mean([res[("last", l, a)][0] for l in [16, 20, 24, 28]], 0), np.mean([res[("last", l, a)][1] for l in [16, 20, 24, 28]], 0))
    ens[("ens_mean_all", a)] = (np.mean([res[("mean", l, a)][0] for l in [16, 20, 24, 28]], 0), np.mean([res[("mean", l, a)][1] for l in [16, 20, 24, 28]], 0))
allr = dict(res)
allr.update(ens)

print("%-24s | %-33s | %-33s" % ("router", "train OOF: fixed / otsu / min", "dev: fixed / otsu / min / q50 / q55"))
for key in allr:
    p_tr, p_dv = allr[key]
    t = tol_thr(p_tr)
    out = []
    for rows, p, pk, unk in [(train, p_tr, pk_tr, u_tr), (dev, p_dv, pk_dv, u_dv)]:
        fixed = bbq_score(route(p, [t] * len(p), pk, unk), rows)
        to = 1 / (1 + np.exp(-by_cat(rows, logit(p), otsu)))
        ot = bbq_score(route(p, to, pk, unk), rows)
        mn = bbq_score(route(p, np.minimum(to, t), pk, unk), rows)
        q50 = bbq_score(route(p, by_cat(rows, p, lambda v: np.quantile(v, 0.5)), pk, unk), rows)
        q55 = bbq_score(route(p, by_cat(rows, p, lambda v: np.quantile(v, 0.45)), pk, unk), rows)
        out.append((fixed, ot, mn, q50, q55))
    name = "%s L%s aug=%s" % (key[0], key[1], key[2]) if len(key) == 3 else "%s aug=%s" % key
    print("%-24s | %.4f / %.4f / %.4f          | %.4f / %.4f / %.4f / %.4f / %.4f" % (name, out[0][0], out[0][1], out[0][2], out[1][0], out[1][1], out[1][2], out[1][3], out[1][4]))

print("\n%-24s | %-12s | %-12s" % ("router", "train OOF q50", "dev q50"))
for key in allr:
    p_tr, p_dv = allr[key]
    a = bbq_score(route(p_tr, by_cat(train, p_tr, lambda v: np.quantile(v, 0.5)), pk_tr, u_tr), train)
    b = bbq_score(route(p_dv, by_cat(dev, p_dv, lambda v: np.quantile(v, 0.5)), pk_dv, u_dv), dev)
    name = "%s L%s aug=%s" % (key[0], key[1], key[2]) if len(key) == 3 else "%s aug=%s" % key
    print("%-24s | %.4f       | %.4f" % (name, a, b))
