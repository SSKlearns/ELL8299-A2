import os
import json
import math
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

base_name = "Qwen/Qwen2.5-1.5B"
inst_name = "Qwen/Qwen2.5-1.5B-Instruct"


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


def get_model(name):
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


def load_setup(out):
    with open(os.path.join(out, "config.json")) as f:
        cfg = json.load(f)
    green = np.load(os.path.join(out, "green.npy"))
    return cfg, green


def z_score(g, w, gamma):
    if len(g) == 0:
        return 0.0
    g = np.array(g, dtype=float)
    w = np.array(w, dtype=float)
    top = (w * (g - gamma)).sum()
    bot = math.sqrt(gamma * (1 - gamma) * (w * w).sum())
    if bot == 0:
        return 0.0
    return float(top / bot)


def model_z(ids, green, q, delta, kind):
    top = 0.0
    bot = 0.0
    for t in range(1, len(ids)):
        g = 1.0 if green[ids[t]] else 0.0
        if kind == "oz":
            w = 1.0 / (1.0 + (math.exp(delta) - 1.0) * q[t])
        else:
            w = 1.0
        top += w * (g - q[t])
        bot += w * w * q[t] * (1 - q[t])
    if bot <= 0:
        return 0.0
    return float(top / math.sqrt(bot))


def rep_skip(ids, k):
    skip = [False] * len(ids)
    if k <= 0:
        return skip
    seen = set()
    for t in range(k, len(ids)):
        ctx = tuple(ids[t - k:t])
        if ctx in seen:
            skip[t] = True
        else:
            seen.add(ctx)
    return skip


def pos_delta(qt, cfg):
    if "q0" in cfg:
        if qt < cfg["q0"]:
            return cfg["lo"]
        return cfg["hi"]
    return cfg["delta"]


def adapt_z(ids, green, q, cfg, kind):
    top = 0.0
    bot = 0.0
    skip = rep_skip(ids, cfg.get("rep_k", 0))
    for t in range(1, len(ids)):
        if skip[t]:
            continue
        d = pos_delta(q[t], cfg)
        if kind == "aoz":
            w = d / (1.0 + (math.exp(d) - 1.0) * q[t])
        else:
            w = d
        g = 1.0 if green[ids[t]] else 0.0
        top += w * (g - q[t])
        bot += w * w * q[t] * (1 - q[t])
    if bot <= 0:
        return 0.0
    return float(top / math.sqrt(bot))


def get_score(ids, green, gamma, kind, ent=None, q=None, delta=0.0):
    g = []
    w = []
    if kind == "mz" or kind == "oz":
        return model_z(ids, green, q, delta, kind)
    if kind == "plain":
        for x in ids:
            g.append(green[x])
            w.append(1.0)
        return z_score(g, w, gamma)
    seen = set()
    for t in range(len(ids)):
        x = ids[t]
        if x in seen:
            continue
        seen.add(x)
        if kind == "weighted":
            if t == 0:
                continue
            w.append(ent[t])
        else:
            w.append(1.0)
        g.append(green[x])
    return z_score(g, w, gamma)


def get_ents(model, all_ids, bs=16, green=None):
    res = []
    qs = []
    gt = None
    if green is not None:
        size = model.get_output_embeddings().weight.shape[0]
        gt = torch.zeros(size, device=model.device)
        gt[:len(green)] = torch.tensor(green, dtype=torch.float32, device=model.device)
    for i in range(0, len(all_ids), bs):
        batch = all_ids[i:i + bs]
        size = max(1, max(len(x) for x in batch))
        ids = torch.zeros(len(batch), size, dtype=torch.long)
        mask = torch.zeros(len(batch), size, dtype=torch.long)
        for j in range(len(batch)):
            n = len(batch[j])
            if n > 0:
                ids[j, :n] = torch.tensor(batch[j])
                mask[j, :n] = 1
        ids = ids.to(model.device)
        mask = mask.to(model.device)
        with torch.no_grad():
            hid = model.model(input_ids=ids, attention_mask=mask).last_hidden_state
        for j in range(len(batch)):
            n = len(batch[j])
            if n == 0:
                res.append([])
                qs.append([])
                continue
            with torch.no_grad():
                logp = torch.log_softmax(model.lm_head(hid[j, :n]).float(), -1)
            prob = logp.exp()
            ent = -(prob * logp).sum(-1).cpu()
            res.append([0.0] + ent[:n - 1].tolist())
            if gt is not None:
                qq = (prob * gt).sum(-1).cpu()
                qs.append([0.0] + qq[:n - 1].tolist())
    if green is not None:
        return res, qs
    return res


def cond_name(path):
    name = os.path.basename(path)
    if name.endswith(".jsonl"):
        name = name[:-6]
    if name.startswith("generated_"):
        name = name[len("generated_"):]
    parts = name.split("_")
    if parts[-1].isdigit():
        parts = parts[:-1]
    if len(parts) == 0:
        return "clean"
    return "_".join(parts)
