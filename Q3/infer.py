import os
import sys
import json
import numpy as np
from common import read, write, get_model, get_feats, make_x, make_x_ood, pick_abstain, load_lr, lr_prob, lr_z, platt, get_dom

inp = sys.argv[1]
out = sys.argv[2]

with open(os.path.join(out, "config.json")) as f:
    cfg = json.load(f)
probe = load_lr(os.path.join(out, "probe.npz"))
calib = load_lr(os.path.join(out, "calib.npz"))
calib_ood = load_lr(os.path.join(out, "calib_ood.npz"))
pool = [[r["question"], r["answer"]] for r in read(os.path.join(out, "pool.jsonl"))]

rows = read(inp)
tok, model = get_model()
feats = get_feats(tok, model, rows, cfg["shots"], pool, k=cfg["k"])
pl = lr_prob(probe, feats[cfg["key"]])
dom = get_dom(feats)
c_two = platt(lr_z(calib, make_x(feats, pl)), dom, cfg["maps"])
c_small = lr_prob(calib_ood, make_x_ood(feats, pl))
conf = np.where(dom == 1, c_small, c_two)
ab = pick_abstain(list(conf))

res = []
for i in range(len(rows)):
    res.append({"id": rows[i]["id"], "answer": feats["answer"][i], "confidence": round(float(conf[i]), 6), "abstain": bool(ab[i])})
dest = os.path.join(out, "pred_" + os.path.basename(inp))
write(dest, res)
print("wrote", dest, "answered", sum(1 for a in ab if not a), "/", len(ab))
