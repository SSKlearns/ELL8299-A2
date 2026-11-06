import os
import sys
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from common import read, clean, get_model, feat_prompt, perm_scores, get_feats, pick_two, bbq_score, cat_thr, save_lr

out = sys.argv[1]
data = sys.argv[2]
only_words = len(sys.argv) > 3 and sys.argv[3] == "words"
os.makedirs(out, exist_ok=True)

rows = read(os.path.join(data, "q1_bbq_train.jsonl"))
print("train rows:", len(rows))

words = sorted(set(clean(r["options"][r["unknown_idx"]]) for r in rows))
with open(os.path.join(out, "unknown_words.json"), "w") as f:
    json.dump(words, f, indent=1)
print("unknown words:", words)

if only_words:
    print("only wrote unknown words, no head trained")
    sys.exit(0)

pool = "mean"
use = [16, 20, 24, 28]
c = 0.1
tol = 0.01
mode = "prior"
min_cat = 20

tok, model = get_model()
print("getting features")
feats = get_feats(tok, model, [feat_prompt(tok, r) for r in rows], use)
print("getting option scores")
scores = perm_scores(tok, model, rows)

y = np.array([1 if r["context_condition"] == "ambig" else 0 for r in rows])
cats = np.array([r["category"] for r in rows])
unk = [r["unknown_idx"] for r in rows]
pick = [pick_two(scores[i], unk[i]) for i in range(len(rows))]
thrs = [round(0.01 * i, 2) for i in range(1, 100)]


def make_head():
    return make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=3000))


def route(p, t):
    return [unk[i] if p[i] >= t[i] else pick[i] for i in range(len(rows))]


print("leave one category out")
p = np.zeros(len(rows))
for l in use:
    x = feats[(pool, l)]
    pl = np.zeros(len(rows))
    for cat in sorted(set(cats)):
        te = cats == cat
        head = make_head()
        head.fit(x[~te], y[~te])
        pl[te] = head.predict_proba(x[te])[:, 1]
    print("layer %d oof acc@0.5 %.4f" % (l, float(np.mean((pl >= 0.5) == (y == 1)))))
    p += pl / len(use)

ss = [bbq_score(route(p, [t] * len(rows)), rows) for t in thrs]
top = max(ss)
thr = thrs[0]
for i in range(len(thrs)):
    if ss[i] >= top - tol:
        thr = thrs[i]
        break
t_cat = cat_thr([r["category"] for r in rows], p, thr, mode, min_cat)
print("fixed thr %.2f oof bbq %.4f" % (thr, bbq_score(route(p, [thr] * len(rows)), rows)))
print("prior (per-category median) oof bbq %.4f" % bbq_score(route(p, t_cat), rows))
print("perfect router oof bbq %.4f" % bbq_score([unk[i] if y[i] == 1 else pick[i] for i in range(len(rows))], rows))
print("two way pick acc on train disambig: %.4f" % float(np.mean([pick[i] == rows[i]["label"] for i in range(len(rows)) if y[i] == 0])))

for j in range(len(use)):
    head = make_head()
    head.fit(feats[(pool, use[j])], y)
    save_lr(os.path.join(out, "head_%d.npz" % j), head)
cfg = {"pool": pool, "layers": use, "c": c, "thr": thr, "mode": mode, "min_cat": min_cat}
with open(os.path.join(out, "config.json"), "w") as f:
    json.dump(cfg, f, indent=1)
print("saved heads to", out, cfg)
