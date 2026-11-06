import os
import collections
import torch
from common import read, get_model, is_right, norm, same, pick_shots, cut

here = os.path.dirname(os.path.abspath(__file__))
data = os.path.join(here, "..", "data", "public")
train = read(os.path.join(data, "q3_qa_train.jsonl"))
ood = read(os.path.join(data, "q3_qa_dev_ood.jsonl"))
test = {"id": train[1000:1500], "ood": ood}

short = "Answer the question with only the answer: a short name, date, number or phrase. Do not explain."
full = "Answer the question with only the answer, but make it complete and specific: the full name of a person or place, the full date, or the precise term. Do not explain."

variants = {
    "short_5shot": (short, pick_shots(train, 5)),
    "full_5shot": (full, pick_shots(train, 5)),
    "full_10shot": (full, pick_shots(train, 10)),
    "short_0shot": (short, []),
}

tok, model = get_model()
stop = set([tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")])


def prompt(sys_text, shots, q):
    msgs = [{"role": "system", "content": sys_text}]
    for a, b in shots:
        msgs.append({"role": "user", "content": a})
        msgs.append({"role": "assistant", "content": b})
    msgs.append({"role": "user", "content": q})
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def run(ps, sample, k=1, bs=32):
    res = []
    for i in range(0, len(ps), bs):
        enc = tok(ps[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        size = enc["input_ids"].shape[1]
        with torch.no_grad():
            if sample:
                o = model.generate(**enc, max_new_tokens=24, do_sample=True, temperature=1.0, top_p=1.0, top_k=0, num_return_sequences=k, pad_token_id=tok.pad_token_id)
            else:
                o = model.generate(**enc, max_new_tokens=32, do_sample=False, temperature=None, top_p=None, top_k=None, pad_token_id=tok.pad_token_id)
        o = o[:, size:].cpu().tolist()
        for s in o:
            res.append(tok.decode(cut(s, stop), skip_special_tokens=True).strip())
    return res


def vote(greedy, samples):
    pool = [greedy] + samples
    best = greedy
    top = 0
    for a in pool:
        n = sum(1 for b in pool if same(a, b))
        if n > top:
            top = n
            best = a
    return best


torch.manual_seed(0)
for name in variants:
    sys_text, shots = variants[name]
    line = []
    for dom in ["id", "ood"]:
        rows = test[dom]
        ps = [prompt(sys_text, shots, r["question"]) for r in rows]
        g = run(ps, False)
        acc = sum(is_right(g[i], rows[i]["answers"]) for i in range(len(rows))) / len(rows)
        words = sum(len(a.split()) for a in g) / len(g)
        line.append("%s acc %.3f (%.1f words)" % (dom, acc, words))
        if name in ["short_5shot", "full_5shot"]:
            s = run(ps, True, k=8)
            v = [vote(g[i], s[i * 8:(i + 1) * 8]) for i in range(len(rows))]
            va = sum(is_right(v[i], rows[i]["answers"]) for i in range(len(rows))) / len(rows)
            line.append("%s vote acc %.3f" % (dom, va))
    print("%-12s | %s" % (name, " | ".join(line)), flush=True)
