import os
import sys
import json
import numpy as np
from transformers import AutoTokenizer, AutoConfig
from common import base_name

out = sys.argv[1]
gamma = 0.25
delta = 1.5
score = "az"
q0 = 0.20
lo = 1.5
hi = 2.25
rep_k = 4
if len(sys.argv) > 2:
    gamma = float(sys.argv[2])
    score = "oz"
    q0 = None
    rep_k = 0
if len(sys.argv) > 3:
    delta = float(sys.argv[3])
if len(sys.argv) > 4:
    score = sys.argv[4]
if len(sys.argv) > 7:
    q0 = float(sys.argv[5])
    lo = float(sys.argv[6])
    hi = float(sys.argv[7])
if len(sys.argv) > 8:
    rep_k = int(sys.argv[8])
key = 15485863

os.makedirs(out, exist_ok=True)
tok = AutoTokenizer.from_pretrained(base_name)
size = AutoConfig.from_pretrained(base_name).vocab_size
n_real = tok.convert_tokens_to_ids("<|endoftext|>")

rng = np.random.default_rng(key)
perm = rng.permutation(n_real)
green = np.zeros(size, dtype=bool)
green[perm[:int(gamma * n_real)]] = True
np.save(os.path.join(out, "green.npy"), green)

cfg = {"key": key, "gamma": gamma, "delta": delta, "score": score, "n_real": int(n_real), "size": int(size), "new_tokens": 200}
if q0 is not None:
    cfg["q0"] = q0
    cfg["lo"] = lo
    cfg["hi"] = hi
if rep_k > 0:
    cfg["rep_k"] = rep_k
with open(os.path.join(out, "config.json"), "w") as f:
    json.dump(cfg, f, indent=1)
print("setup done:", cfg, "green tokens:", int(green.sum()))
