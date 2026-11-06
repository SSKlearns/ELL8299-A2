import os
import sys
import pickle
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from common import read, get_model, pick_shots, make_prompt, cut, sys_text, same, norm, layers

if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.4)

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
files = ["q3_qa_train", "q3_qa_dev", "q3_qa_dev_ood"]
rows = []
for f in files:
    rows += read("../data/public/" + f + ".jsonl")
n = len(rows)
qs = [r["question"] for r in rows]
tok, model = get_model()
stop = set([tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")])
yes = tok.encode("Yes", add_special_tokens=False)[0]
no = tok.encode("No", add_special_tokens=False)[0]
bs = 32
k = 8

shots = pick_shots(rows[:2000])
vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(qs)
mat = vec.transform(qs)
sim = (mat @ mat.T).toarray()
np.fill_diagonal(sim, -1)
knn = []
knn_sim = []
for i in range(n):
    order = np.argsort(-sim[i])[:5]
    knn.append([[qs[j], rows[j]["answers"][0]] for j in order[::-1]])
    knn_sim.append(float(sim[i, order[0]]))


def zero_prompt(q):
    msgs = [{"role": "system", "content": sys_text}, {"role": "user", "content": q}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def greedy(prompts):
    ans = []
    ids_all = []
    lp_mean = []
    lp_min = []
    lp_first = []
    for i in range(0, len(prompts), bs):
        enc = tok(prompts[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        size = enc["input_ids"].shape[1]
        with torch.no_grad():
            o = model.generate(**enc, max_new_tokens=32, do_sample=False, repetition_penalty=1.0, temperature=None, top_p=None, top_k=None,
                               output_scores=True, return_dict_in_generate=True, pad_token_id=tok.pad_token_id)
        seq = o.sequences[:, size:].cpu().tolist()
        logp = torch.stack([torch.log_softmax(s.float(), -1) for s in o.scores], 1).cpu()
        for j in range(len(seq)):
            t = cut(seq[j], stop)
            ids_all.append(t)
            ans.append(tok.decode(t, skip_special_tokens=True).strip())
            v = [logp[j, m, t[m]].item() for m in range(len(t))]
            if len(v) == 0:
                v = [-20.0]
            lp_mean.append(float(np.mean(v)))
            lp_min.append(float(np.min(v)))
            lp_first.append(v[0])
    return ans, ids_all, lp_mean, lp_min, lp_first


def sample(prompts):
    res = []
    for i in range(0, len(prompts), bs):
        enc = tok(prompts[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        size = enc["input_ids"].shape[1]
        with torch.no_grad():
            o = model.generate(**enc, max_new_tokens=24, do_sample=True, temperature=1.0, top_p=1.0, top_k=0, repetition_penalty=1.0,
                               num_return_sequences=k, pad_token_id=tok.pad_token_id)
        s = o[:, size:].cpu().tolist()
        for j in range(len(s) // k):
            res.append([tok.decode(cut(s[j * k + m], stop), skip_special_tokens=True).strip() for m in range(k)])
    return res


def yes_no(prompts):
    res = []
    for i in range(0, len(prompts), bs):
        enc = tok(prompts[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            lg = model(**enc).logits[:, -1, :].float()
        lp = torch.log_softmax(lg, -1)
        res += (lp[:, yes] - lp[:, no]).cpu().tolist()
    return res


def check_prompt(q, a, cands=None):
    text = "Question: " + q + "\n"
    if cands is not None:
        text += "Some possible answers: " + "; ".join(cands) + "\n"
    text += "Proposed answer: " + a + "\nIs the proposed answer correct? Reply with only Yes or No."
    msgs = [{"role": "system", "content": "You are a careful fact checker."}, {"role": "user", "content": text}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def hidden(prompts, ids_all):
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
            o = model.model(input_ids=ids, attention_mask=mask, position_ids=pos, output_hidden_states=True)
        for j in range(len(seqs)):
            for l in layers:
                h = o.hidden_states[l][j].float()
                hp[l].append(h[size - 1 - lens[j]].cpu().numpy())
                ha[l].append(h[size - 1].cpu().numpy())
    res = {}
    for l in layers:
        res["hp" + str(l)] = np.array(hp[l])
        res["ha" + str(l)] = np.array(ha[l])
    return res


def sem_ent(ss):
    cl = []
    for s in ss:
        for c in cl:
            if same(s, c[0]):
                c.append(s)
                break
        else:
            cl.append([s])
    p = np.array([len(c) for c in cl]) / len(ss)
    return float(-(p * np.log(p)).sum())


torch.manual_seed(0)
zero, _, _, _, _ = greedy([zero_prompt(q) for q in qs])
print("zero done")
res = {"zero": zero, "knn_sim": knn_sim, "shots": shots}
for mode in ["fixed", "knn"]:
    if mode == "fixed":
        prompts = [make_prompt(tok, q, shots) for q in qs]
    else:
        prompts = [make_prompt(tok, qs[i], knn[i]) for i in range(n)]
    torch.manual_seed(0)
    ans, ids_all, lp_mean, lp_min, lp_first = greedy(prompts)
    samp = sample(prompts)
    print(mode, "generation done")
    f = {"answer": ans, "lp_mean": lp_mean, "lp_min": lp_min, "lp_first": lp_first, "n_tok": [len(t) for t in ids_all], "samples": samp,
         "agree": [sum(1 for s in samp[i] if same(s, ans[i])) / k for i in range(n)],
         "uniq": [len(set(norm(s) for s in samp[i])) / k for i in range(n)],
         "sem_ent": [sem_ent(samp[i]) for i in range(n)],
         "has_q": [1.0 if q.strip().endswith("?") else 0.0 for q in qs],
         "upper": [1.0 if q.strip()[:1].isupper() else 0.0 for q in qs]}
    f["ptrue"] = yes_no([check_prompt(qs[i], ans[i]) for i in range(n)])
    f["ptrue_b"] = yes_no([check_prompt(qs[i], ans[i], samp[i]) for i in range(n)])
    f.update(hidden(prompts, ids_all))
    res[mode] = f
    print(mode, "features done")
with open(os.path.join(out, "sig2.pkl"), "wb") as fh:
    pickle.dump(res, fh)
print("saved")
