import sys
import random
import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from common import read, get_model, pick_shots, make_prompt, cut, is_right

if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.4)

target = sys.argv[1]
epochs = int(sys.argv[2])
lr = float(sys.argv[3])
train = read("../data/public/q3_qa_train.jsonl")
dev = read("../data/public/q3_qa_dev.jsonl")
ood = read("../data/public/q3_qa_dev_ood.jsonl")
tok, model = get_model()
shots = pick_shots(train)
stop = set([tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")])
end = tok.convert_tokens_to_ids("<|im_end|>")


def acc(rows, bs=32):
    model.eval()
    right = 0
    for i in range(0, len(rows), bs):
        batch = rows[i:i + bs]
        enc = tok([make_prompt(tok, r["question"], shots) for r in batch], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        size = enc["input_ids"].shape[1]
        with torch.no_grad():
            o = model.generate(**enc, max_new_tokens=32, do_sample=False, temperature=None, top_p=None, top_k=None, pad_token_id=tok.pad_token_id)
        for j, s in enumerate(o[:, size:].cpu().tolist()):
            right += is_right(tok.decode(cut(s, stop), skip_special_tokens=True).strip(), batch[j]["answers"])
    return right / len(rows)


print("before: dev %.4f ood %.4f" % (acc(dev), acc(ood)))
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
data = []
for r in train:
    if target == "longest":
        a = max(r["answers"], key=len)
    else:
        a = r["answers"][0]
    p = tok(make_prompt(tok, r["question"], shots), add_special_tokens=False)["input_ids"]
    t = tok(a, add_special_tokens=False)["input_ids"] + [end]
    data.append((p, t))
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
random.seed(0)
torch.manual_seed(0)
bs = 16
steps = epochs * (len(data) // bs)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 20) * max(0.0, 1 - s / steps))
step = 0
for ep in range(epochs):
    model.train()
    random.shuffle(data)
    for i in range(0, len(data) - bs + 1, bs):
        for mb in range(0, bs, 4):
            batch = data[i + mb:i + mb + 4]
            size = max(len(p) + len(t) for p, t in batch)
            ids = torch.full((len(batch), size), tok.pad_token_id, dtype=torch.long)
            lab = torch.full((len(batch), size), -100, dtype=torch.long)
            mask = torch.zeros(len(batch), size, dtype=torch.long)
            for j, (p, t) in enumerate(batch):
                s = p + t
                ids[j, size - len(s):] = torch.tensor(s)
                lab[j, size - len(t):] = torch.tensor(t)
                mask[j, size - len(s):] = 1
            ids = ids.to(model.device)
            lab = lab.to(model.device)
            mask = mask.to(model.device)
            pos = (mask.cumsum(-1) - 1).clamp(min=0)
            loss = model(input_ids=ids, attention_mask=mask, position_ids=pos, labels=lab).loss / 4
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad()
        step += 1
        if step % 50 == 0:
            print("epoch", ep, "step", step, "loss %.4f" % loss.item())
    print("after epoch %d: dev %.4f ood %.4f" % (ep, acc(dev), acc(ood)))
