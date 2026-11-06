import re
import json
import string
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.feature_extraction.text import TfidfVectorizer

name = "Qwen/Qwen2.5-1.5B-Instruct"
sys_text = "Answer the question with only the answer: a short name, date, number or phrase. Do not explain."
check_text = "You are a careful fact checker."
layers = [14, 20, 26]


def read(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def get_model():
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"
    if torch.cuda.is_available():
        dev = "cuda"
        dt = torch.bfloat16
    else:
        dev = "cpu"
        dt = torch.float32
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dt)
    model.to(dev)
    model.eval()
    return tok, model


def norm(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def is_right(ans, golds):
    a = " " + norm(ans) + " "
    for g in golds:
        g = norm(g)
        if len(g) > 0 and (" " + g + " ") in a:
            return 1
    return 0


def same(a, b):
    a = norm(a)
    b = norm(b)
    if len(a) == 0 or len(b) == 0:
        return False
    return a == b or (" " + a + " ") in (" " + b + " ") or (" " + b + " ") in (" " + a + " ")


def pick_shots(rows, n=5):
    shots = []
    seen = set()
    for r in rows:
        w = r["question"].split()[0]
        a = r["answers"][0]
        if w in seen or len(a.split()) > 3 or len(r["answers"]) > 1:
            continue
        seen.add(w)
        shots.append([r["question"], a])
        if len(shots) == n:
            break
    return shots


def make_prompt(tok, q, shots):
    msgs = [{"role": "system", "content": sys_text}]
    for a, b in shots:
        msgs.append({"role": "user", "content": a})
        msgs.append({"role": "assistant", "content": b})
    msgs.append({"role": "user", "content": q})
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def check_prompt(tok, q, a, cands=None):
    text = "Question: " + q + "\n"
    if cands is not None:
        text += "Some possible answers: " + "; ".join(cands) + "\n"
    text += "Proposed answer: " + a + "\nIs the proposed answer correct? Reply with only Yes or No."
    msgs = [{"role": "system", "content": check_text}, {"role": "user", "content": text}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def get_knn(pool, qs, n=5):
    pq = [p[0] for p in pool]
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(pq)
    sim = (vec.transform(qs) @ vec.transform(pq).T).toarray()
    keys = {}
    for j in range(len(pq)):
        k = norm(pq[j])
        if k not in keys:
            keys[k] = []
        keys[k].append(j)
    res = []
    for i in range(len(qs)):
        for j in keys.get(norm(qs[i]), []):
            sim[i, j] = -1.0
        order = np.argsort(-sim[i], kind="stable")[:n]
        res.append([pool[j] for j in order[::-1]])
    return res


def cut(ids, stop):
    res = []
    for x in ids:
        if x in stop:
            break
        res.append(x)
    return res


def get_stop(tok):
    return set([tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")])


def greedy(tok, model, prompts, bs):
    stop = get_stop(tok)
    ans = []
    ids_all = []
    lps = []
    for i in range(0, len(prompts), bs):
        enc = tok(prompts[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        size = enc["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=32, do_sample=False, repetition_penalty=1.0, temperature=None, top_p=None, top_k=None,
                                 output_scores=True, return_dict_in_generate=True, pad_token_id=tok.pad_token_id)
        seq = out.sequences[:, size:].cpu().tolist()
        logp = torch.stack([torch.log_softmax(s.float(), -1) for s in out.scores], 1).cpu()
        for j in range(len(seq)):
            t = cut(seq[j], stop)
            ids_all.append(t)
            ans.append(tok.decode(t, skip_special_tokens=True).strip())
            v = [logp[j, n, t[n]].item() for n in range(len(t))]
            if len(v) == 0:
                v = [-20.0]
            lps.append(v)
    return ans, ids_all, lps


def sample(tok, model, prompts, k, bs):
    stop = get_stop(tok)
    res = []
    for i in range(0, len(prompts), bs):
        enc = tok(prompts[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        size = enc["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=24, do_sample=True, temperature=1.0, top_p=1.0, top_k=0, repetition_penalty=1.0,
                                 num_return_sequences=k, pad_token_id=tok.pad_token_id)
        s = out[:, size:].cpu().tolist()
        for j in range(len(s) // k):
            res.append([tok.decode(cut(s[j * k + m], stop), skip_special_tokens=True).strip() for m in range(k)])
    return res


def yes_no(tok, model, prompts, bs):
    yes = tok.encode("Yes", add_special_tokens=False)[0]
    no = tok.encode("No", add_special_tokens=False)[0]
    res = []
    for i in range(0, len(prompts), bs):
        enc = tok(prompts[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        pos = (enc["attention_mask"].cumsum(-1) - 1).clamp(min=0)
        with torch.no_grad():
            lg = model(**enc, position_ids=pos, logits_to_keep=1).logits[:, -1, :].float()
        lp = torch.log_softmax(lg, -1)
        res += (lp[:, yes] - lp[:, no]).cpu().tolist()
    return res


def hidden(tok, model, prompts, ids_all, bs):
    hp = {}
    ha = {}
    for l in layers:
        hp[l] = []
        ha[l] = []
    for i in range(0, len(prompts), bs):
        seqs = []
        lens = []
        for j in range(i, min(i + bs, len(prompts))):
            p = tok(prompts[j], add_special_tokens=False)["input_ids"]
            seqs.append(p + ids_all[j])
            lens.append(len(ids_all[j]))
        size = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), size), tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros(len(seqs), size, dtype=torch.long)
        for j in range(len(seqs)):
            ids[j, size - len(seqs[j]):] = torch.tensor(seqs[j])
            mask[j, size - len(seqs[j]):] = 1
        ids = ids.to(model.device)
        mask = mask.to(model.device)
        pos = (mask.cumsum(-1) - 1).clamp(min=0)
        with torch.no_grad():
            out = model.model(input_ids=ids, attention_mask=mask, position_ids=pos, output_hidden_states=True)
        for j in range(len(seqs)):
            for l in layers:
                h = out.hidden_states[l][j].float()
                hp[l].append(h[size - 1 - lens[j]].cpu().numpy())
                ha[l].append(h[size - 1].cpu().numpy())
    res = {}
    for l in layers:
        res["hp" + str(l)] = np.array(hp[l])
        res["ha" + str(l)] = np.array(ha[l])
    return res


def get_feats(tok, model, rows, shots, pool, k=8, bs=32, seed=0):
    qs = [r["question"] for r in rows]
    torch.manual_seed(seed)
    prompts = [make_prompt(tok, q, shots) for q in qs]
    ans, ids_all, lps = greedy(tok, model, prompts, bs)
    samp = sample(tok, model, prompts, k, bs)
    print("main answers and samples done")
    zero_p = [tok.apply_chat_template([{"role": "system", "content": sys_text}, {"role": "user", "content": q}], tokenize=False, add_generation_prompt=True) for q in qs]
    zero, _, _ = greedy(tok, model, zero_p, bs)
    knn = get_knn(pool, qs)
    knn_p = [make_prompt(tok, qs[i], knn[i]) for i in range(len(qs))]
    other, _, _ = greedy(tok, model, knn_p, bs)
    other_s = sample(tok, model, knn_p, k, bs)
    print("second-opinion answers done")
    pt = yes_no(tok, model, [check_prompt(tok, qs[i], ans[i]) for i in range(len(qs))], bs)
    ptb = yes_no(tok, model, [check_prompt(tok, qs[i], ans[i], samp[i]) for i in range(len(qs))], bs)
    print("self-check done")
    res = {"answer": ans,
           "lp_mean": [float(np.mean(v)) for v in lps],
           "lp_min": [float(np.min(v)) for v in lps],
           "lp_first": [v[0] for v in lps],
           "n_tok": [len(t) for t in ids_all],
           "agree": [sum(1 for s in samp[i] if same(s, ans[i])) / k for i in range(len(qs))],
           "uniq": [len(set(norm(s) for s in samp[i])) / k for i in range(len(qs))],
           "has_q": [1.0 if q.strip().endswith("?") else 0.0 for q in qs],
           "upper": [1.0 if q.strip()[:1].isupper() else 0.0 for q in qs],
           "ptrue": pt,
           "ptrue_b": ptb,
           "ag_zero": [1.0 if same(ans[i], zero[i]) else 0.0 for i in range(len(qs))],
           "ag_other": [1.0 if same(ans[i], other[i]) else 0.0 for i in range(len(qs))],
           "ag_os": [sum(1 for s in other_s[i] if same(s, ans[i])) / k for i in range(len(qs))]}
    res.update(hidden(tok, model, prompts, ids_all, bs))
    return res


def get_dom(f):
    return np.array([1 if f["has_q"][i] > 0 or f["upper"][i] > 0 else 0 for i in range(len(f["has_q"]))])


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def make_x(f, pl):
    cols = [f["lp_mean"], f["lp_min"], f["lp_first"], np.log1p(f["n_tok"]), f["agree"], f["uniq"], f["has_q"], f["upper"],
            f["ptrue"], f["ptrue_b"], f["ag_zero"], f["ag_other"], f["ag_os"], logit(pl)]
    return np.array(cols, dtype=np.float64).T


def make_x_ood(f, pl):
    cols = [f["lp_mean"], f["lp_min"], f["lp_first"], np.log1p(f["n_tok"]), f["agree"], f["uniq"], f["has_q"], f["upper"], logit(pl)]
    return np.array(cols, dtype=np.float64).T


def pick_abstain(conf, sims=4000):
    conf = np.array(conf, dtype=np.float64)
    if len(conf) == 0:
        return []
    rng = np.random.default_rng(0)
    order = np.argsort(-conf, kind="stable")
    c = conf[order]
    nn = np.arange(1, len(c) + 1)
    total = np.zeros(len(c))
    for t in range(0, sims, 500):
        draws = rng.random((min(500, sims - t), len(c))) < c
        risk = np.cumsum(~draws, 1) / nn
        total += (nn / len(c) * np.clip(1 - (risk - 0.2) / 0.1, 0, 1)).sum(0)
    best = int(np.argmax(total)) + 1
    ab = [True] * len(c)
    for i in order[:best]:
        ab[i] = False
    return ab


def save_lr(path, pipe):
    sc = pipe.named_steps["standardscaler"]
    lr = pipe.named_steps["logisticregression"]
    np.savez(path, mean=sc.mean_, scale=sc.scale_, coef=lr.coef_[0], bias=lr.intercept_)


def load_lr(path):
    z = np.load(path)
    return {"mean": z["mean"], "scale": z["scale"], "coef": z["coef"], "bias": float(z["bias"][0])}


def lr_z(h, x):
    return ((np.asarray(x, dtype=np.float64) - h["mean"]) / h["scale"]) @ h["coef"] + h["bias"]


def lr_prob(h, x):
    return 1.0 / (1.0 + np.exp(-lr_z(h, x)))


def platt(z, dom, maps):
    p = np.zeros(len(z))
    for i in range(len(z)):
        a, b = maps[str(int(dom[i]))]
        p[i] = 1.0 / (1.0 + np.exp(-(a * z[i] + b)))
    return p


def auroc(c, y):
    pos = [c[i] for i in range(len(y)) if y[i] == 1]
    neg = [c[i] for i in range(len(y)) if y[i] == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    good = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                good += 1
            elif p == n:
                good += 0.5
    return good / (len(pos) * len(neg))


def ece(c, y):
    bins = {}
    for i in range(len(c)):
        b = 0
        while b < 9 and c[i] > (b + 1) / 10:
            b += 1
        if b not in bins:
            bins[b] = []
        bins[b].append(i)
    total = 0.0
    for b in bins:
        ix = bins[b]
        total += len(ix) / len(c) * abs(sum(y[i] for i in ix) / len(ix) - sum(c[i] for i in ix) / len(ix))
    return total


def selective(y, ab):
    ans = [i for i in range(len(y)) if not ab[i]]
    cov = len(ans) / len(y)
    if len(ans) == 0:
        return 0.0, cov, 0.0
    risk = sum(1 - y[i] for i in ans) / len(ans)
    if risk <= 0.2:
        return cov, cov, risk
    return cov * max(0.0, 1 - (risk - 0.2) / 0.1), cov, risk


def q3_score(c, y, ab):
    a = auroc(c, y)
    e = ece(c, y)
    s, cov, risk = selective(y, ab)
    return (a + (1 - e) + s) / 3, a, e, s, cov, risk
