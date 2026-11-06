import os
import sys
import json
import pickle
import random
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from common import read, pick_two, bbq_score, cat_thr

here = os.path.dirname(os.path.abspath(__file__))
data = os.path.join(here, "..", "data", "public")
cache = os.path.join(here, "..", "out", "q1_exp")
dev_name = "cuda" if torch.cuda.is_available() else "cpu"

train = read(os.path.join(data, "q1_bbq_train.jsonl"))
dev = read(os.path.join(data, "q1_bbq_dev.jsonl"))
f_tr = pickle.load(open(os.path.join(cache, "f_train.pkl"), "rb"))
f_dv = pickle.load(open(os.path.join(cache, "f_dev.pkl"), "rb"))
s_tr = np.array(pickle.load(open(os.path.join(cache, "s_train.pkl"), "rb")))
s_dv = np.array(pickle.load(open(os.path.join(cache, "s_dev.pkl"), "rb")))
use = [16, 20, 24, 28]
tol = 0.01
thrs = [round(0.01 * i, 2) for i in range(1, 100)]


def info(rows, s):
    y = np.array([1 if r["context_condition"] == "ambig" else 0 for r in rows])
    cats = np.array([r["category"] for r in rows])
    tpl = np.array([r["template_id"] for r in rows])
    unk = [r["unknown_idx"] for r in rows]
    pick = [pick_two(s[i], unk[i]) for i in range(len(rows))]
    ln = np.array([len(r["context"].split()) for r in rows])
    return y, cats, tpl, unk, pick, ln


y_tr, c_tr, t_tr, u_tr, k_tr, l_tr = info(train, s_tr)
y_dv, c_dv, t_dv, u_dv, k_dv, l_dv = info(dev, s_dv)


def route(p, t, unk, pick):
    return [unk[i] if p[i] >= t[i] else pick[i] for i in range(len(p))]


def supcon(z, y, tau, grp):
    z = F.normalize(z, dim=1)
    sim = z @ z.T / tau
    n = len(z)
    ok = ~torch.eye(n, dtype=torch.bool, device=z.device)
    if grp is not None:
        ok = ok & (grp[:, None] == grp[None, :])
    sim = sim.masked_fill(~ok, -1e9)
    logp = sim - torch.logsumexp(sim, 1, keepdim=True)
    pos = ok & (y[:, None] == y[None, :])
    cnt = pos.sum(1)
    keep = cnt > 0
    loss = -(logp * pos).sum(1)[keep] / cnt[keep]
    return loss.mean()


def make_pairs(y, tpl, n_max, seed):
    rng = random.Random(seed)
    a = []
    d = []
    for t in sorted(set(tpl)):
        ia = [i for i in range(len(y)) if tpl[i] == t and y[i] == 1]
        idd = [i for i in range(len(y)) if tpl[i] == t and y[i] == 0]
        for i in ia:
            for j in idd:
                a.append(i)
                d.append(j)
    z = list(zip(a, d))
    rng.shuffle(z)
    z = z[:n_max]
    return [p[0] for p in z], [p[1] for p in z]


def fit_lr(x, y, xt):
    head = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))
    head.fit(x, y)
    return head.predict_proba(xt)[:, 1]


def fit_lin(x, y, tpl, xt, lam, seed):
    torch.manual_seed(seed)
    mu = x.mean(0)
    sd = x.std(0) + 1e-6
    xa = torch.tensor((x - mu) / sd, dtype=torch.float32, device=dev_name)
    xb = torch.tensor((xt - mu) / sd, dtype=torch.float32, device=dev_name)
    ya = torch.tensor(y, dtype=torch.float32, device=dev_name)
    w = torch.zeros(x.shape[1], device=dev_name, requires_grad=True)
    b = torch.zeros(1, device=dev_name, requires_grad=True)
    pa, pd = make_pairs(y, tpl, 20000, seed)
    pa = torch.tensor(pa, device=dev_name)
    pd = torch.tensor(pd, device=dev_name)
    n = len(y)
    opt = torch.optim.LBFGS([w, b], lr=1, max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        s = xa @ w + b
        loss = F.binary_cross_entropy_with_logits(s, ya) + (w * w).sum() / (2 * 0.1 * n)
        if lam > 0:
            loss = loss + lam * F.softplus(s[pd] - s[pa]).mean()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        return torch.sigmoid(xb @ w + b).cpu().numpy()


def fit_mlp(x, y, tpl, xt, lam, kind, seed):
    torch.manual_seed(seed)
    mu = x.mean(0)
    sd = x.std(0) + 1e-6
    xa = torch.tensor((x - mu) / sd, dtype=torch.float32, device=dev_name)
    xb = torch.tensor((xt - mu) / sd, dtype=torch.float32, device=dev_name)
    ya = torch.tensor(y, dtype=torch.float32, device=dev_name)
    tmap = {}
    for t in tpl:
        if t not in tmap:
            tmap[t] = len(tmap)
    ga = torch.tensor([tmap[t] for t in tpl], device=dev_name)
    enc = torch.nn.Sequential(torch.nn.Dropout(0.2), torch.nn.Linear(x.shape[1], 256), torch.nn.GELU()).to(dev_name)
    cls = torch.nn.Linear(256, 1).to(dev_name)
    proj = torch.nn.Linear(256, 64).to(dev_name)
    ps = list(enc.parameters()) + list(cls.parameters()) + list(proj.parameters())
    opt = torch.optim.AdamW(ps, lr=1e-3, weight_decay=0.05)
    for ep in range(200):
        enc.train()
        h = enc(xa)
        s = cls(h)[:, 0]
        loss = F.binary_cross_entropy_with_logits(s, ya)
        if lam > 0:
            grp = ga if kind == "tpl" else None
            loss = loss + lam * supcon(proj(h), ya, 0.1, grp)
        opt.zero_grad()
        loss.backward()
        opt.step()
    enc.eval()
    with torch.no_grad():
        return torch.sigmoid(cls(enc(xb))[:, 0]).cpu().numpy()


def fit(name, x, y, tpl, xt):
    if name == "lr":
        return fit_lr(x, y, xt)
    if name.startswith("lin"):
        lam = float(name.split("_")[1])
        return fit_lin(x, y, tpl, xt, lam, 0)
    kind = name.split("_")[0]
    lam = float(name.split("_")[1])
    p = np.zeros(len(xt))
    for seed in range(3):
        p += fit_mlp(x, y, tpl, xt, lam, kind, seed) / 3
    return p


def cat_auc(p, y, cats):
    v = []
    for c in sorted(set(cats)):
        m = cats == c
        v.append(roc_auc_score(y[m], p[m]))
    return v


def pick_thr(p):
    ss = [bbq_score(route(p, [t] * len(p), u_tr, k_tr), train) for t in thrs]
    top = max(ss)
    for i in range(len(thrs)):
        if ss[i] >= top - tol:
            return thrs[i]
    return thrs[0]


def wrong(p, t, y):
    a = int(sum(1 for i in range(len(p)) if y[i] == 1 and p[i] < t[i]))
    d = int(sum(1 for i in range(len(p)) if y[i] == 0 and p[i] >= t[i]))
    return a, d


names = sys.argv[1:] if len(sys.argv) > 1 else ["lr", "lin_0", "lin_1", "lin_5", "mlp_0", "all_0.5", "all_2", "tpl_0.5", "tpl_2"]
cats_list = sorted(set(c_tr))
res = {}
for name in names:
    oof = np.zeros(len(train))
    pd_ = np.zeros(len(dev))
    for l in use:
        x = f_tr[("mean", l)]
        xd = f_dv[("mean", l)]
        po = np.zeros(len(train))
        for cat in cats_list:
            te = c_tr == cat
            po[te] = fit(name, x[~te], y_tr[~te], t_tr[~te], x[te])
        oof += po / len(use)
        pd_ += fit(name, x, y_tr, t_tr, xd) / len(use)
    thr = pick_thr(oof)
    tm_tr = cat_thr(list(c_tr), oof, thr, "prior", 20)
    tm_dv = cat_thr(list(c_dv), pd_, thr, "prior", 20)
    tf_tr = [thr] * len(train)
    tf_dv = [thr] * len(dev)
    r = {}
    r["loco_auc"] = cat_auc(oof, y_tr, c_tr)
    r["dev_auc"] = cat_auc(pd_, y_dv, c_dv)
    r["thr"] = thr
    r["loco_fixed"] = bbq_score(route(oof, tf_tr, u_tr, k_tr), train)
    r["loco_median"] = bbq_score(route(oof, tm_tr, u_tr, k_tr), train)
    r["dev_fixed"] = bbq_score(route(pd_, tf_dv, u_dv, k_dv), dev)
    r["dev_median"] = bbq_score(route(pd_, tm_dv, u_dv, k_dv), dev)
    r["loco_wrong_median"] = wrong(oof, tm_tr, y_tr)
    r["dev_wrong_fixed"] = wrong(pd_, tf_dv, y_dv)
    m = y_dv == 1
    r["dev_len_corr"] = float(np.corrcoef(l_dv[m], pd_[m])[0, 1])
    res[name] = r
    print("%-8s locoAUC %.4f (min %.3f) devAUC %.4f (min %.3f) | thr %.2f | LOCO fixed %.4f median %.4f | dev fixed %.4f median %.4f | LOCO median wrong (amb->dis, dis->amb) %s | dev fixed wrong %s | dev len corr %.2f" % (
        name, np.mean(r["loco_auc"]), min(r["loco_auc"]), np.mean(r["dev_auc"]), min(r["dev_auc"]), thr,
        r["loco_fixed"], r["loco_median"], r["dev_fixed"], r["dev_median"], r["loco_wrong_median"], r["dev_wrong_fixed"], r["dev_len_corr"]), flush=True)
    print("         loco auc per cat", dict(zip(cats_list, [round(v, 3) for v in r["loco_auc"]])), flush=True)

perfect_tr = bbq_score([u_tr[i] if y_tr[i] == 1 else k_tr[i] for i in range(len(train))], train)
perfect_dv = bbq_score([u_dv[i] if y_dv[i] == 1 else k_dv[i] for i in range(len(dev))], dev)
print("perfect router: LOCO %.4f dev %.4f" % (perfect_tr, perfect_dv))
with open(os.path.join(cache, "contrast_res.json"), "w") as f:
    json.dump(res, f, indent=1)
