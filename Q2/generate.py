import os
import sys
import torch
from common import read, write, get_model, load_setup, base_name

inp = sys.argv[1]
out = sys.argv[2]
seed = int(sys.argv[3])
bs = 32

cfg, green = load_setup(out)
rows = read(inp)
tok, model = get_model(base_name)
dev = model.device

size = model.get_output_embeddings().weight.shape[0]
gvec = torch.zeros(size, device=dev)
gvec[:len(green)] = torch.tensor(green, dtype=torch.float32, device=dev)
bad = torch.zeros(size, device=dev)
bad[cfg["n_real"]:] = -float("inf")


k = cfg.get("rep_k", 0)


def next_probs(logits, on):
    if "q0" in cfg:
        p = torch.softmax(logits + bad, -1)
        q = (p * gvec).sum(-1, keepdim=True)
        d = torch.where(q < cfg["q0"], cfg["lo"], cfg["hi"])
    else:
        d = cfg["delta"]
    if k > 0:
        d = d * on
    return torch.softmax(logits + d * gvec + bad, -1)

gen = torch.Generator(device=dev)
gen.manual_seed(seed)
torch.manual_seed(seed)

res = []
for i in range(0, len(rows), bs):
    batch = rows[i:i + bs]
    enc = tok([r["prompt"] for r in batch], return_tensors="pt", padding=True, add_special_tokens=False)
    ids = enc["input_ids"].to(dev)
    mask = enc["attention_mask"].to(dev)
    pos = (mask.cumsum(-1) - 1).clamp(min=0)
    with torch.no_grad():
        o = model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=True)
    past = o.past_key_values
    logits = o.logits[:, -1, :].float()
    new = []
    hist = [[] for j in range(len(batch))]
    seen = [set() for j in range(len(batch))]
    for step in range(cfg["new_tokens"]):
        on = torch.ones(len(batch), 1, device=dev)
        if k > 0:
            for j in range(len(batch)):
                if len(hist[j]) >= k:
                    ctx = tuple(hist[j][-k:])
                    if ctx in seen[j]:
                        on[j, 0] = 0.0
                    else:
                        seen[j].add(ctx)
        probs = next_probs(logits, on)
        nxt = torch.multinomial(probs, 1, generator=gen)
        new.append(nxt)
        if k > 0:
            got = nxt[:, 0].tolist()
            for j in range(len(batch)):
                hist[j].append(int(got[j]))
        if step == cfg["new_tokens"] - 1:
            break
        mask = torch.cat([mask, torch.ones_like(nxt)], 1)
        pos = pos[:, -1:] + 1
        with torch.no_grad():
            o = model(input_ids=nxt, attention_mask=mask, position_ids=pos, past_key_values=past, use_cache=True)
        past = o.past_key_values
        logits = o.logits[:, -1, :].float()
    new = torch.cat(new, 1).cpu().tolist()
    for j in range(len(batch)):
        t = [int(x) for x in new[j]]
        res.append({"id": batch[j]["id"], "text": tok.decode(t, skip_special_tokens=True), "token_ids": t})
    print("generated", len(res), "/", len(rows))

dest = os.path.join(out, "generated_" + str(seed) + ".jsonl")
write(dest, res)
print("wrote", dest)
