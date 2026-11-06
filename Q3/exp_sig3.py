import os
import sys
import pickle
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from common import read, get_model, is_right, layers

if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.4)

out = sys.argv[1]
files = ["q3_qa_train", "q3_qa_dev", "q3_qa_dev_ood"]
rows = []
for f in files:
    rows += read("../data/public/" + f + ".jsonl")
n = len(rows)
qs = [r["question"] for r in rows]
with open(os.path.join(out, "sig2.pkl"), "rb") as fh:
    sig = pickle.load(fh)
ans = sig["fixed"]["answer"]
samp = sig["fixed"]["samples"]
tok, model = get_model()
yes = tok.encode("Yes", add_special_tokens=False)[0]
no = tok.encode("No", add_special_tokens=False)[0]
bs = 16

vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(qs)
mat = vec.transform(qs)
sim = (mat @ mat.T).toarray()
np.fill_diagonal(sim, -1)
knn = []
for i in range(n):
    order = np.argsort(-sim[i])[:5]
    knn.append([[qs[j], rows[j]["answers"][0]] for j in order[::-1]])

good = [i for i in range(2000) if is_right(ans[i], rows[i]["answers"]) and len(ans[i]) > 0][:3]
bad = [i for i in range(2000) if not is_right(ans[i], rows[i]["answers"]) and len(ans[i]) > 0][:3]
demo = []
for a, b in zip(good, bad):
    demo += [(b, "No"), (a, "Yes")]
print("demo ids", demo)


def body(i, ref=None):
    text = ""
    if ref is not None:
        text += "Related questions with their correct answers:\n"
        for a, b in ref:
            text += "- " + a + " -> " + b + "\n"
        text += "\n"
    text += "Question: " + qs[i] + "\n"
    text += "Some possible answers: " + "; ".join(samp[i]) + "\n"
    text += "Proposed answer: " + ans[i] + "\nIs the proposed answer correct? Reply with only Yes or No."
    return text


def chat(msgs):
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


sys_msg = {"role": "system", "content": "You are a careful fact checker."}
p_plain = [chat([sys_msg, {"role": "user", "content": body(i)}]) for i in range(n)]
p_knn = [chat([sys_msg, {"role": "user", "content": body(i, knn[i])}]) for i in range(n)]
p_few = []
for i in range(n):
    msgs = [sys_msg]
    for j, lab in demo:
        msgs.append({"role": "user", "content": body(j)})
        msgs.append({"role": "assistant", "content": lab})
    msgs.append({"role": "user", "content": body(i)})
    p_few.append(chat(msgs))


def run(prompts, keep):
    res = []
    hs = {}
    for l in layers:
        hs[l] = []
    for i in range(0, len(prompts), bs):
        enc = tok(prompts[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            o = model(**enc, output_hidden_states=keep, logits_to_keep=1)
        lp = torch.log_softmax(o.logits[:, -1, :].float(), -1)
        res += (lp[:, yes] - lp[:, no]).cpu().tolist()
        if keep:
            for l in layers:
                hs[l].append(o.hidden_states[l][:, -1, :].float().cpu().numpy())
    if keep:
        return res, {l: np.concatenate(hs[l]) for l in layers}
    return res, None


res = {"demo": demo}
v, h = run(p_plain, True)
res["ptrue_b"] = v
for l in layers:
    res["hv" + str(l)] = h[l]
print("plain done")
res["ptrue_k"], _ = run(p_knn, False)
print("knn done")
res["ptrue_f"], _ = run(p_few, False)
print("few done")
with open(os.path.join(out, "sig3.pkl"), "wb") as fh:
    pickle.dump(res, fh)
print("saved")
