import os
import sys
import json
import numpy as np
from common import read, write, get_model, find_unk, qa_prompt, feat_prompt, get_scores, perm_scores, get_feats, pick_two, cat_thr, load_lr, lr_prob

mode = sys.argv[1]
out = sys.argv[2]
inp = sys.argv[3]
dest = sys.argv[4]

if mode != "bias" and mode != "qa":
    print("mode must be bias or qa, got " + mode)
    sys.exit(1)

rows = read(inp)
print("rows:", len(rows))
res = []

if mode == "qa":
    tok, model = get_model()
    scores = get_scores(tok, model, [qa_prompt(tok, r) for r in rows], [len(r["options"]) for r in rows], bs=8)
    for i in range(len(rows)):
        res.append({"id": rows[i]["id"], "pred_idx": int(np.argmax(scores[i]))})
else:
    words_path = os.path.join(out, "unknown_words.json")
    if os.path.exists(words_path):
        with open(words_path) as f:
            words = set(json.load(f))
    else:
        print("no unknown_words.json found, using word overlap only")
        words = set()
    unk = [find_unk(r, words) for r in rows]
    if os.path.exists(os.path.join(out, "head_0.npz")):
        with open(os.path.join(out, "config.json")) as f:
            cfg = json.load(f)
        head = [load_lr(os.path.join(out, "head_%d.npz" % j)) for j in range(len(cfg["layers"]))]
        tok, model = get_model()
        feats = get_feats(tok, model, [feat_prompt(tok, r) for r in rows], cfg["layers"])
        p = np.zeros(len(rows))
        for j in range(len(cfg["layers"])):
            p += lr_prob(head[j], feats[(cfg["pool"], cfg["layers"][j])]) / len(cfg["layers"])
        t = cat_thr([r["category"] for r in rows], p, cfg["thr"], cfg["mode"], cfg["min_cat"])
        scores = perm_scores(tok, model, rows)
        cnt = 0
        for i in range(len(rows)):
            if p[i] >= t[i]:
                k = unk[i]
                cnt += 1
            else:
                k = pick_two(scores[i], unk[i])
            res.append({"id": rows[i]["id"], "pred_idx": int(k)})
        print("called ambiguous: %d / %d (mode %s, fixed thr %.2f)" % (cnt, len(rows), cfg["mode"], cfg["thr"]))
    else:
        print("no head_0.npz found, answering unknown for every item")
        for i in range(len(rows)):
            res.append({"id": rows[i]["id"], "pred_idx": int(unk[i])})

write(dest, res)
print("wrote", dest)
