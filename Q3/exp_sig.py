import os
import sys
import pickle
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from common import read, get_model, pick_shots, make_prompt, cut, sys_text

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

shots = pick_shots(rows[:2000])
alt = pick_shots(rows[1000:2000])
vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(qs)
mat = vec.transform(qs)
sim = (mat @ mat.T).toarray()
np.fill_diagonal(sim, -1)
knn = []
for i in range(n):
    order = np.argsort(-sim[i])[:5]
    knn.append([[qs[j], rows[j]["answers"][0]] for j in order[::-1]])


def zero_prompt(q):
    msgs = [{"role": "system", "content": sys_text}, {"role": "user", "content": q}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def greedy(prompts, max_new=32):
    res = []
    for i in range(0, len(prompts), bs):
        enc = tok(prompts[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        size = enc["input_ids"].shape[1]
        with torch.no_grad():
            o = model.generate(**enc, max_new_tokens=max_new, do_sample=False, temperature=None, top_p=None, top_k=None, pad_token_id=tok.pad_token_id)
        for s in o[:, size:].cpu().tolist():
            res.append(tok.decode(cut(s, stop), skip_special_tokens=True).strip())
    return res


def sample(prompts, k=8, temp=1.0):
    res = []
    for i in range(0, len(prompts), bs):
        enc = tok(prompts[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        size = enc["input_ids"].shape[1]
        with torch.no_grad():
            o = model.generate(**enc, max_new_tokens=24, do_sample=True, temperature=temp, top_p=1.0, top_k=0, num_return_sequences=k, pad_token_id=tok.pad_token_id)
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


res = {}
torch.manual_seed(0)
main_p = [make_prompt(tok, q, shots) for q in qs]
res["main"] = greedy(main_p)
print("main done")
res["samples"] = sample(main_p)
print("samples done")
res["zero"] = greedy([zero_prompt(q) for q in qs])
print("zero done")
res["alt"] = greedy([make_prompt(tok, q, alt) for q in qs])
print("alt done")
res["knn"] = greedy([make_prompt(tok, qs[i], knn[i]) for i in range(n)])
print("knn done")
res["ptrue"] = yes_no([check_prompt(qs[i], res["main"][i]) for i in range(n)])
res["ptrue_b"] = yes_no([check_prompt(qs[i], res["main"][i], res["samples"][i]) for i in range(n)])
print("ptrue done")
res["shots"] = shots
res["alt_shots"] = alt
with open(os.path.join(out, "sig.pkl"), "wb") as fh:
    pickle.dump(res, fh)
print("saved")
