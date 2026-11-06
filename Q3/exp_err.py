import re
import sys
import string
import pickle
import collections
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from common import read, is_right, norm, make_x, q3_score, pick_abstain

out = "../out/q3"
data = "../data/public"
files = ["q3_qa_train", "q3_qa_dev", "q3_qa_dev_ood"]
rows = []
src = []
feats = {}
for i, f in enumerate(files):
    r = read(data + "/" + f + ".jsonl")
    rows += r
    src += [i] * len(r)
    with open(out + "/feats_" + f + ".pkl", "rb") as fh:
        feats[f] = pickle.load(fh)
src = np.array(src)
all_f = {}
for key in feats[files[0]]:
    if type(feats[files[0]][key]) is list:
        all_f[key] = sum([feats[f][key] for f in files], [])
    else:
        all_f[key] = np.concatenate([feats[f][key] for f in files])
ans = all_f["answer"]
y = np.array([is_right(a, r["answers"]) for a, r in zip(ans, rows)])
ood = src == 2

folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=0).split(np.zeros(len(y)), src * 2 + y))


def oof(x, c, w=None):
    p = np.zeros(len(y))
    for tr, te in folds:
        h = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=3000))
        if w is None:
            h.fit(x[tr], y[tr])
        else:
            h.fit(x[tr], y[tr], logisticregression__sample_weight=w[tr])
        p[te] = h.predict_proba(x[te])[:, 1]
    return p


pl = oof(all_f["hp14"], 0.05)
conf = oof(make_x(all_f, pl, "plain"), 1.0, np.where(ood, 5.0, 1.0))
for nm, m in [("train+dev (id)", src < 2), ("dev", src == 1), ("dev_ood", ood)]:
    sc, a, e, s, cov, risk = q3_score(list(conf[m]), list(y[m]), pick_abstain(list(conf[m]), 0.0))
    print("%-15s n %4d acc %.3f AUROC %.4f ECE %.4f set %.4f" % (nm, m.sum(), y[m].mean(), a, e, sc))


def norm2(s):
    s = s.lower()
    s = "".join(" " if (ch in string.punctuation or not (ch.isalnum() or ch.isspace())) else ch for ch in s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def right2(a, golds):
    a = " " + norm2(a) + " "
    for g in golds:
        g = norm2(g)
        if len(g) > 0 and (" " + g + " ") in a:
            return 1
    return 0


y2 = np.array([right2(a, r["answers"]) for a, r in zip(ans, rows)])
print("\nnormalisation that turns punctuation into spaces: labels changed %d / %d (0->1 %d, 1->0 %d)" % (
    (y2 != y).sum(), len(y), ((y == 0) & (y2 == 1)).sum(), ((y == 1) & (y2 == 0)).sum()))

print("\n=== error types among wrong answers")
for nm, m in [("id", src < 2), ("ood", ood)]:
    kinds = collections.Counter()
    for i in np.where(m & (y == 0))[0]:
        a = norm(ans[i])
        gs = [norm(g) for g in rows[i]["answers"]]
        if len(a) == 0:
            kinds["empty"] += 1
        elif any((" " + a + " ") in (" " + g + " ") for g in gs):
            kinds["answer is part of a gold answer (too short)"] += 1
        elif any(len(set(a.split()) & set(g.split())) > 0 for g in gs):
            kinds["shares a word with gold"] += 1
        else:
            kinds["no overlap"] += 1
    n = (m & (y == 0)).sum()
    print(nm, "wrong", n)
    for k, v in kinds.most_common():
        print("   %-45s %4d  %.1f%%" % (k, v, 100 * v / n))

print("\n=== accuracy by confidence rank (id = train+dev, OOF)")
for nm, m in [("id", src < 2), ("ood", ood)]:
    c = conf[m]
    yy = y[m]
    order = np.argsort(-c)
    for frac in [0.01, 0.02, 0.05, 0.1, 0.2]:
        n = max(1, int(round(frac * len(c))))
        print("   %-4s top %4.0f%% (n %4d): accuracy %.3f  mean conf %.3f" % (nm, 100 * frac, n, yy[order[:n]].mean(), c[order[:n]].mean()))

print("\n=== accuracy by agreement of 8 samples with greedy")
ag = np.array(all_f["agree"])
for nm, m in [("id", src < 2), ("ood", ood)]:
    for lo, hi in [(0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)]:
        mm = m & (ag >= lo) & (ag < hi)
        if mm.sum() > 0:
            print("   %-4s agree [%.2f, %.2f): n %4d acc %.3f mean conf %.3f" % (nm, lo, hi, mm.sum(), y[mm].mean(), conf[mm].mean()))

print("\n=== OOD by question template and popularity")
tmpl = []
for r in rows:
    q = r["question"]
    t = re.sub(r"(is|of|for|was|in|by) .*\?$", r"\1 X?", q)
    tmpl.append(t)
for t, n in collections.Counter(tmpl[i] for i in np.where(ood)[0]).most_common():
    m = ood & np.array([x == t for x in tmpl])
    print("   %-45s n %3d acc %.3f mean conf %.3f" % (t[:45], n, y[m].mean(), conf[m].mean()))
pq = np.array([r.get("popularity_quartile", -1) for r in rows])
for q in [1, 2, 3, 4]:
    m = ood & (pq == q)
    print("   quartile %d n %3d acc %.3f mean conf %.3f auroc %.3f" % (q, m.sum(), y[m].mean(), conf[m].mean(), roc_auc_score(y[m], conf[m]) if 0 < y[m].sum() < m.sum() else float("nan")))

print("\n=== most confident wrong answers (id)")
ix = [i for i in np.argsort(-conf) if src[i] < 2 and y[i] == 0][:25]
for i in ix:
    print("   %.2f ag %.2f | %-55s | ans %-30s | gold %s" % (conf[i], ag[i], rows[i]["question"][:55], ans[i][:30], rows[i]["answers"][:3]))
print("\n=== least confident right answers (id)")
ix = [i for i in np.argsort(conf) if src[i] < 2 and y[i] == 1][:15]
for i in ix:
    print("   %.3f ag %.2f | %-55s | ans %-30s | gold %s" % (conf[i], ag[i], rows[i]["question"][:55], ans[i][:30], rows[i]["answers"][:3]))
print("\n=== most confident wrong answers (ood)")
ix = [i for i in np.argsort(-conf) if ood[i] and y[i] == 0][:15]
for i in ix:
    print("   %.2f ag %.2f | %-50s | ans %-30s | gold %s" % (conf[i], ag[i], rows[i]["question"][:50], ans[i][:30], rows[i]["answers"][:3]))

print("\n=== calibration by bin (OOF)")
for nm, m in [("id", src < 2), ("ood", ood)]:
    for b in range(10):
        mm = m & (conf > b / 10) & (conf <= (b + 1) / 10) if b > 0 else m & (conf <= 0.1)
        if mm.sum() > 0:
            print("   %-4s bin %d n %4d mean conf %.3f acc %.3f" % (nm, b, mm.sum(), conf[mm].mean(), y[mm].mean()))
