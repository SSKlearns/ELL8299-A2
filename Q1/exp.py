import os
import re
import json
import random
import pickle
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from common import read, clean, get_model, find_unk, bias_prompt, feat_prompt, get_scores, get_feats, pick_two, bbq_score

here = os.path.dirname(os.path.abspath(__file__))
data = os.path.join(here, "..", "data", "public")
cache = os.path.join(here, "..", "out", "q1_exp")
os.makedirs(cache, exist_ok=True)

train = read(os.path.join(data, "q1_bbq_train.jsonl"))
dev = read(os.path.join(data, "q1_bbq_dev.jsonl"))
words = set(clean(r["options"][r["unknown_idx"]]) for r in train)
use = [16, 20, 24, 28]


def sents(t):
    return re.split(r"(?<=[.!?])\s+", t.strip())


rng = random.Random(0)
dis = [r for r in train if r["context_condition"] == "disambig"]
aug = []
for r in train:
    j = rng.choice(dis)
    while j["category"] == r["category"]:
        j = rng.choice(dis)
    x = dict(r)
    x["context"] = r["context"] + " " + sents(j["context"])[-1]
    aug.append(x)

tok, model = get_model()


def cached(name, fn):
    path = os.path.join(cache, name + ".pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    x = fn()
    with open(path, "wb") as f:
        pickle.dump(x, f)
    return x


def get_cloze(rows, bs=16):
    pairs = []
    for r in rows:
        p = tok(bias_prompt(tok, r), add_special_tokens=False)["input_ids"]
        for o in r["options"]:
            a = tok(o, add_special_tokens=False)["input_ids"]
            pairs.append((p, a))
    sc = []
    fe = {}
    for l in use:
        fe[l] = []
    for i in range(0, len(pairs), bs):
        batch = pairs[i:i + bs]
        size = max(len(p) + len(a) for p, a in batch)
        ids = torch.full((len(batch), size), tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros(len(batch), size, dtype=torch.long)
        for j in range(len(batch)):
            p, a = batch[j]
            ids[j, :len(p) + len(a)] = torch.tensor(p + a)
            mask[j, :len(p) + len(a)] = 1
        ids = ids.to(model.device)
        mask = mask.to(model.device)
        with torch.no_grad():
            out = model.model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
        for j in range(len(batch)):
            p, a = batch[j]
            pos = torch.arange(len(p) - 1, len(p) + len(a) - 1, device=model.device)
            tgt = torch.tensor(a, device=model.device)
            with torch.no_grad():
                logp = torch.log_softmax(model.lm_head(out.last_hidden_state[j, pos]).float(), -1)
            sc.append(logp[torch.arange(len(a), device=model.device), tgt].mean().item())
            for l in use:
                fe[l].append(out.hidden_states[l][j, len(p) + len(a) - 1].float().cpu().numpy())
    n = len(rows[0]["options"])
    sc = np.array(sc).reshape(len(rows), n)
    for l in use:
        fe[l] = np.array(fe[l]).reshape(len(rows), n, -1)
    return sc, fe


print("features")
f_tr = cached("f_train", lambda: get_feats(tok, model, [feat_prompt(tok, r) for r in train], use))
f_dv = cached("f_dev", lambda: get_feats(tok, model, [feat_prompt(tok, r) for r in dev], use))
f_ag = cached("f_aug", lambda: get_feats(tok, model, [feat_prompt(tok, r) for r in aug], use))
print("letter scores")
s_tr = np.array(cached("s_train", lambda: get_scores(tok, model, [bias_prompt(tok, r) for r in train], [3] * len(train))))
s_dv = np.array(cached("s_dev", lambda: get_scores(tok, model, [bias_prompt(tok, r) for r in dev], [3] * len(dev))))
print("cloze scores")
c_tr, h_tr = cached("c_train", lambda: get_cloze(train))
c_dv, h_dv = cached("c_dev", lambda: get_cloze(dev))

y_tr = np.array([r["context_condition"] == "ambig" for r in train])
y_dv = np.array([r["context_condition"] == "ambig" for r in dev])
cat_tr = np.array([r["category"] for r in train])
cat_dv = np.array([r["category"] for r in dev])
u_tr = [find_unk(r, words) for r in train]
u_dv = [find_unk(r, words) for r in dev]
thrs = [round(0.01 * i, 2) for i in range(1, 100)]


def head():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))


def route(p, pk, unk, t):
    return [unk[i] if p[i] >= t else pk[i] for i in range(len(p))]


def pick_thr(p, pk, tol=0.01):
    ss = [bbq_score(route(p, pk, u_tr, t), train) for t in thrs]
    top = max(ss)
    for i in range(len(thrs)):
        if ss[i] >= top - tol:
            return thrs[i], ss[i]


def router(x_tr, x_dv, x_ag=None):
    p = np.zeros(len(train))
    for c in sorted(set(cat_tr)):
        te = cat_tr == c
        xs = x_tr[~te]
        ys = y_tr[~te]
        if x_ag is not None:
            xs = np.concatenate([xs, x_ag[~te]])
            ys = np.concatenate([ys, y_tr[~te]])
        h = head()
        h.fit(xs, ys)
        p[te] = h.predict_proba(x_tr[te])[:, 1]
    xs = x_tr
    ys = y_tr
    if x_ag is not None:
        xs = np.concatenate([x_tr, x_ag])
        ys = np.concatenate([y_tr, y_tr])
    h = head()
    h.fit(xs, ys)
    return p, h.predict_proba(x_dv)[:, 1]


def two(s, unk):
    return [pick_two(list(s[i]), unk[i]) for i in range(len(s))]


def dis_acc(pk, rows, unk):
    ok = [pk[i] == rows[i]["label"] for i in range(len(rows)) if rows[i]["context_condition"] == "disambig"]
    return float(np.mean(ok))


print("\n===== pickers (acc on disambig items, dev bbq with perfect router)")
prior = np.exp(s_tr - s_tr.max(1, keepdims=True))
prior = (prior / prior.sum(1, keepdims=True)).mean(0)
print("letter position prior from train:", np.round(prior, 3))
pickers = {}
pickers["letter"] = (two(s_tr, u_tr), two(s_dv, u_dv))
pickers["letter_prior"] = (two(s_tr - np.log(prior), u_tr), two(s_dv - np.log(prior), u_dv))
pickers["cloze"] = (two(c_tr, u_tr), two(c_dv, u_dv))
pickers["letter+cloze"] = (two(s_tr - np.log(prior) + c_tr, u_tr), two(s_dv - np.log(prior) + c_dv, u_dv))

dmask = ~y_tr
for l in use:
    xs = []
    ys = []
    gs = []
    for i in np.where(dmask)[0]:
        for k in range(3):
            if k == u_tr[i]:
                continue
            xs.append(h_tr[l][i, k])
            ys.append(int(k == train[i]["label"]))
            gs.append(cat_tr[i])
    xs = np.array(xs)
    ys = np.array(ys)
    gs = np.array(gs)
    oof = np.zeros(len(ys))
    for c in sorted(set(gs)):
        te = gs == c
        h = head()
        h.fit(xs[~te], ys[~te])
        oof[te] = h.predict_proba(xs[te])[:, 1]
    sc_tr = np.full((len(train), 3), -9.0)
    j = 0
    for i in np.where(dmask)[0]:
        for k in range(3):
            if k == u_tr[i]:
                continue
            sc_tr[i, k] = oof[j]
            j += 1
    h = head()
    h.fit(xs, ys)
    sc_dv = np.full((len(dev), 3), -9.0)
    for i in range(len(dev)):
        for k in range(3):
            if k == u_dv[i]:
                continue
            sc_dv[i, k] = h.predict_proba(h_dv[l][i, k][None])[0, 1]
    pk_tr = two(sc_tr, u_tr)
    pk_tr = [pk_tr[i] if dmask[i] else pickers["letter"][0][i] for i in range(len(train))]
    pickers["probe_L" + str(l)] = (pk_tr, two(sc_dv, u_dv))

for name in pickers:
    pk_tr, pk_dv = pickers[name]
    best_dv = bbq_score([u_dv[i] if y_dv[i] else pk_dv[i] for i in range(len(dev))], dev)
    print("%-14s train_dis_acc %.4f  dev_dis_acc %.4f  dev_bbq_perfect_router %.4f" % (name, dis_acc(pk_tr, train, u_tr), dis_acc(pk_dv, dev, u_dv), best_dv))

print("\n===== routers (picker = letter), threshold = lowest within 0.01 of best OOF bbq")
pk_tr, pk_dv = pickers["letter"]
res = {}
for pool in ["last", "mean"]:
    for l in use:
        for a in [False, True]:
            key = (pool, l)
            p_tr, p_dv = router(f_tr[key], f_dv[key], f_ag[key] if a else None)
            t, s = pick_thr(p_tr, pk_tr)
            d = bbq_score(route(p_dv, pk_dv, u_dv, t), dev)
            res[(pool, l, a)] = (p_tr, p_dv)
            print("%-4s L%d aug=%-5s oof_bbq %.4f thr %.2f | dev_auc %.4f dev_amb_rec %.3f dev_dis_as_amb %.3f dev_bbq %.4f" % (pool, l, a, s, t, roc_auc_score(y_dv, p_dv), (p_dv[y_dv] >= t).mean(), (p_dv[~y_dv] >= t).mean(), d))

for a in [False, True]:
    for name, ls in [("last_ens_20_24_28", [20, 24, 28]), ("last_ens_all", use)]:
        p_tr = np.mean([res[("last", l, a)][0] for l in ls], 0)
        p_dv = np.mean([res[("last", l, a)][1] for l in ls], 0)
        t, s = pick_thr(p_tr, pk_tr)
        d = bbq_score(route(p_dv, pk_dv, u_dv, t), dev)
        print("%-17s aug=%-5s oof_bbq %.4f thr %.2f | dev_auc %.4f dev_amb_rec %.3f dev_dis_as_amb %.3f dev_bbq %.4f" % (name, a, s, t, roc_auc_score(y_dv, p_dv), (p_dv[y_dv] >= t).mean(), (p_dv[~y_dv] >= t).mean(), d))

print("\n===== length check: corr(context words, p_ambig) on dev ambig items (last L28)")
for a in [False, True]:
    p_dv = res[("last", 28, a)][1]
    ln = np.array([len(r["context"].split()) for r in dev])
    print("aug=%s corr %.3f" % (a, np.corrcoef(ln[y_dv], p_dv[y_dv])[0, 1]))

with open(os.path.join(cache, "exp_res.pkl"), "wb") as f:
    pickle.dump({"res": res, "pickers": pickers, "u_dv": u_dv, "u_tr": u_tr}, f)
print("saved")
