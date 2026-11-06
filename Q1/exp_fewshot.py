import os
import sys
import pickle
import random
import numpy as np
from common import read, get_model, letters, get_scores, get_orders, pick_two, bbq_score

here = os.path.dirname(os.path.abspath(__file__))
data = os.path.join(here, "..", "data", "public")
cache = os.path.join(here, "..", "out", "q1_exp")

train = read(os.path.join(data, "q1_bbq_train.jsonl"))
dev = read(os.path.join(data, "q1_bbq_dev.jsonl"))
f_tr = pickle.load(open(os.path.join(cache, "f_train.pkl"), "rb"))
f_dv = pickle.load(open(os.path.join(cache, "f_dev.pkl"), "rb"))
use = [16, 20, 24, 28]
n_shot = 6
nums = "123456789"


def stack(f):
    return np.concatenate([f[("mean", l)] for l in use], 1)


x_tr = stack(f_tr)
x_dv = stack(f_dv)
mu = x_tr.mean(0)
sd = x_tr.std(0) + 1e-6


def unit(x):
    z = (x - mu) / sd
    return z / np.linalg.norm(z, axis=1, keepdims=True)


z_tr = unit(x_tr)
z_dv = unit(x_dv)
bank = [i for i in range(len(train)) if train[i]["context_condition"] == "disambig"]


def is_st(r):
    return r["label"] == r["biased_idx"]


def can_add(chosen, j, cat_cap):
    r = train[j]
    for k in chosen:
        if train[k]["template_id"] == r["template_id"]:
            return False
    same_cat = sum(1 for k in chosen if train[k]["category"] == r["category"])
    pol = sum(1 for k in chosen if train[k]["question_polarity"] == r["question_polarity"])
    st = sum(1 for k in chosen if is_st(train[k]) == is_st(r))
    return same_cat < cat_cap and pol < 3 and st < 3


def select(order, cat_cap):
    chosen = []
    for j in order:
        if can_add(chosen, j, cat_cap):
            chosen.append(j)
            if len(chosen) == n_shot:
                break
    return chosen


rng = random.Random(0)
shuffled = bank[:]
rng.shuffle(shuffled)
fixed_cache = {}


def fixed_demos(skip_cat):
    if skip_cat not in fixed_cache:
        fixed_cache[skip_cat] = select([j for j in shuffled if train[j]["category"] != skip_cat], 2)
    return fixed_cache[skip_cat]


def knn_demos(z, skip_cat):
    cand = [j for j in bank if train[j]["category"] != skip_cat]
    sims = z_tr[cand] @ z
    order = [cand[k] for k in np.argsort(-sims)]
    return select(order, n_shot)


def item_text(r, order, labs=letters, word="letter"):
    text = r["context"] + "\n" + r["question"] + "\n"
    for i in range(len(order)):
        text += labs[i] + ". " + r["options"][order[i]] + "\n"
    text += "Answer with the " + word + " of the correct option."
    return text


def demo_msgs(demos, shift):
    msgs = []
    for k in range(len(demos)):
        r = train[demos[k]]
        g = r["label"]
        u = r["unknown_idx"]
        o = [i for i in range(3) if i != g and i != u][0]
        gs = (k + shift) % 3
        us = (gs + 1 + k // 3) % 3
        order = [0, 0, 0]
        order[gs] = g
        order[us] = u
        order[3 - gs - us] = o
        msgs.append({"role": "user", "content": item_text(r, order, nums, "number")})
        msgs.append({"role": "assistant", "content": nums[gs]})
    return msgs


def score(tok, model, rows, idx, demo_fn, name):
    path = os.path.join(cache, "fs_%s.pkl" % name)
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    prompts = []
    for i in idx:
        demos = demo_fn(i)
        pre = demo_msgs(demos, i)
        for o in get_orders(3):
            msgs = pre + [{"role": "user", "content": item_text(rows[i], o)}]
            prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
    if name.startswith("knn_dev") or name.startswith("fixed_dev"):
        print("example prompt for", name, ":\n" + prompts[0][-1800:], flush=True)
    lens = [len(p) for p in prompts]
    srt = sorted(range(len(prompts)), key=lambda k: lens[k])
    raw_s = get_scores(tok, model, [prompts[k] for k in srt], [3] * len(prompts), bs=16)
    raw = [None] * len(prompts)
    for a in range(len(srt)):
        raw[srt[a]] = raw_s[a]
    res = []
    k = 0
    for i in idx:
        tot = np.zeros(3)
        for o in get_orders(3):
            v = np.array(raw[k], dtype=np.float64)
            v = v - v.max()
            v = v - np.log(np.exp(v).sum())
            for j in range(3):
                tot[o[j]] += v[j]
            k += 1
        res.append(list(tot / 6))
    pickle.dump(res, open(path, "wb"))
    return res


def report(name, rows, idx, s):
    pick = {}
    for k in range(len(idx)):
        pick[idx[k]] = pick_two(s[k], rows[idx[k]]["unknown_idx"])
    ok = {i: pick[i] == rows[i]["label"] for i in idx}
    nonneg = [i for i in idx if rows[i]["question_polarity"] == "nonneg"]
    neg = [i for i in idx if rows[i]["question_polarity"] == "neg"]
    anti = [i for i in idx if not is_st(rows[i])]
    full = []
    for i in range(len(rows)):
        if rows[i]["context_condition"] == "ambig":
            full.append(rows[i]["unknown_idx"])
        else:
            full.append(pick[i])
    print("%-14s dis acc %.4f | neg %.3f nonneg %.3f anti-stereo %.3f | BBQ (perfect router) %.4f" % (
        name, np.mean([ok[i] for i in idx]), np.mean([ok[i] for i in neg]), np.mean([ok[i] for i in nonneg]),
        np.mean([ok[i] for i in anti]), bbq_score(full, rows)), flush=True)
    cats = sorted(set(rows[i]["category"] for i in idx))
    print("               per cat", {c: round(float(np.mean([ok[i] for i in idx if rows[i]["category"] == c])), 3) for c in cats}, flush=True)


tok, model = get_model()
i_tr = [i for i in range(len(train)) if train[i]["context_condition"] == "disambig"]
i_dv = [i for i in range(len(dev)) if dev[i]["context_condition"] == "disambig"]
variants = sys.argv[1:] if len(sys.argv) > 1 else ["zero", "fixed", "knn"]
for v in variants:
    if v == "zero":
        fd = lambda i: []
        ft = lambda i: []
    elif v == "fixed":
        fd = lambda i: fixed_demos(None)
        ft = lambda i: fixed_demos(train[i]["category"])
    else:
        fd = lambda i: knn_demos(z_dv[i], None)
        ft = lambda i: knn_demos(z_tr[i], train[i]["category"])
    s = score(tok, model, dev, i_dv, fd, v + "_dev")
    report(v + " dev", dev, i_dv, s)
    s = score(tok, model, train, i_tr, ft, v + "_train")
    report(v + " train", train, i_tr, s)
