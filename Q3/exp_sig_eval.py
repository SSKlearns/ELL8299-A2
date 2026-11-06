import sys
import pickle
import collections
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from common import read, is_right, norm, same, make_x, q3_score, pick_abstain

files = ["q3_qa_train", "q3_qa_dev", "q3_qa_dev_ood"]
rows = []
src = []
feats = {}
for i, f in enumerate(files):
    r = read("../data/public/" + f + ".jsonl")
    rows += r
    src += [i] * len(r)
    with open("../out/q3/feats_" + f + ".pkl", "rb") as fh:
        feats[f] = pickle.load(fh)
src = np.array(src)
all_f = {}
for key in feats[files[0]]:
    if type(feats[files[0]][key]) is list:
        all_f[key] = sum([feats[f][key] for f in files], [])
    else:
        all_f[key] = np.concatenate([feats[f][key] for f in files])
with open(sys.argv[1], "rb") as fh:
    sig = pickle.load(fh)
ood = src == 2
n = len(rows)
golds = [r["answers"] for r in rows]

print("main answer equal to cached v1 answer: %d / %d" % (sum(1 for a, b in zip(sig["main"], all_f["answer"]) if a == b), n))
print("\n=== accuracy of each answer source")
for k in ["main", "zero", "alt", "knn"]:
    yy = np.array([is_right(a, g) for a, g in zip(sig[k], golds)])
    print("   %-5s id(train+dev) %.4f  dev %.4f  ood %.4f" % (k, yy[src < 2].mean(), yy[src == 1].mean(), yy[ood].mean()))
vote = []
for i in range(n):
    cands = [sig["main"][i], sig["zero"][i], sig["alt"][i], sig["knn"][i]]
    best = max(cands, key=lambda a: sum(1 for b in cands if same(a, b)))
    vote.append(best)
yy = np.array([is_right(a, g) for a, g in zip(vote, golds)])
print("   %-5s id(train+dev) %.4f  dev %.4f  ood %.4f" % ("vote4", yy[src < 2].mean(), yy[src == 1].mean(), yy[ood].mean()))
pool = []
for i in range(n):
    cands = [sig["main"][i]] + sig["samples"][i]
    best = max(cands, key=lambda a: sum(1 for b in cands if same(a, b)))
    pool.append(any(is_right(a, golds[i]) for a in cands + [sig["zero"][i], sig["alt"][i], sig["knn"][i]]))
pool = np.array(pool)
print("   any of the 12 answers correct (oracle): id %.4f ood %.4f" % (pool[src < 2].mean(), pool[ood].mean()))

y = np.array([is_right(a, g) for a, g in zip(sig["main"], golds)])
ag_zero = np.array([1.0 if same(a, b) else 0.0 for a, b in zip(sig["main"], sig["zero"])])
ag_alt = np.array([1.0 if same(a, b) else 0.0 for a, b in zip(sig["main"], sig["alt"])])
ag_knn = np.array([1.0 if same(a, b) else 0.0 for a, b in zip(sig["main"], sig["knn"])])
ent = []
for i in range(n):
    cands = [norm(s) for s in sig["samples"][i]]
    cl = []
    for s in cands:
        for c in cl:
            if same(s, c[0]):
                c.append(s)
                break
        else:
            cl.append([s])
    p = np.array([len(c) for c in cl]) / len(cands)
    ent.append(float(-(p * np.log(p)).sum()))
ent = np.array(ent)
new = {"ptrue": np.array(sig["ptrue"]), "ptrue_b": np.array(sig["ptrue_b"]), "ag_zero": ag_zero, "ag_alt": ag_alt, "ag_knn": ag_knn,
       "ag_prompts": ag_zero + ag_alt + ag_knn, "sem_ent": ent}
print("\n=== single-signal AUROC (main answer correctness)")
base = {"agree(8 samples)": np.array(all_f["agree"]), "exp mean logprob": np.exp(np.array(all_f["lp_mean"]))}
base.update(new)
for k, v in base.items():
    sgn = -1 if k == "sem_ent" else 1
    print("   %-18s id %.4f  ood %.4f" % (k, roc_auc_score(y[src < 2], sgn * v[src < 2]), roc_auc_score(y[ood], sgn * v[ood])))


def oof(x, c, w, seed):
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(np.zeros(n), src * 2 + y))
    p = np.zeros(n)
    for tr, te in folds:
        h = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=3000))
        if w is None:
            h.fit(x[tr], y[tr])
        else:
            h.fit(x[tr], y[tr], logisticregression__sample_weight=w[tr])
        p[te] = h.predict_proba(x[te])[:, 1]
    return p


w = np.where(ood, 5.0, 1.0)
pls = [oof(all_f["hp14"], 0.05, None, sd) for sd in range(3)]


def evaluate(name, extra):
    out = []
    for sd in range(3):
        x = make_x(all_f, pls[sd], "plain")
        if len(extra) > 0:
            x = np.column_stack([x] + [new[k] for k in extra])
        p = oof(x, 1.0, w, sd)
        r = []
        for m in [src < 2, src == 1, ood]:
            sc, a, e, s, cov, risk = q3_score(list(p[m]), list(y[m]), pick_abstain(list(p[m]), 0.0))
            r += [a, e, (a + 1 - e) / 3]
        out.append(r)
    o = np.mean(out, 0)
    print("%-40s id AUROC %.4f ECE %.4f | dev AUROC %.4f | ood AUROC %.4f ECE %.4f | set w/o selective: id %.4f ood %.4f mean %.4f" % (
        name, o[0], o[1], o[3], o[6], o[7], o[2], o[8], (o[2] + o[8]) / 2))


print("\n=== calibrator with extra signals (OOF, 3 fold seeds)")
evaluate("v1 features", [])
evaluate("+ ptrue", ["ptrue"])
evaluate("+ ptrue_b", ["ptrue_b"])
evaluate("+ prompt agreement (3)", ["ag_zero", "ag_alt", "ag_knn"])
evaluate("+ semantic entropy", ["sem_ent"])
evaluate("+ ptrue_b + prompt agreement", ["ptrue_b", "ag_zero", "ag_alt", "ag_knn"])
evaluate("+ all", ["ptrue", "ptrue_b", "ag_zero", "ag_alt", "ag_knn", "sem_ent"])
