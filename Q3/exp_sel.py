import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from common import read, is_right, make_x, selective, pick_abstain

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
y = np.array([is_right(a, r["answers"]) for a, r in zip(all_f["answer"], rows)])
ood = src == 2


def oof(x, c, w, seed):
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(np.zeros(len(y)), src * 2 + y))
    p = np.zeros(len(y))
    for tr, te in folds:
        h = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=3000))
        if w is None:
            h.fit(x[tr], y[tr])
        else:
            h.fit(x[tr], y[tr], logisticregression__sample_weight=w[tr])
        p[te] = h.predict_proba(x[te])[:, 1]
    return p


def exp_pick(conf, sims=4000, seed=0):
    rng = np.random.default_rng(seed)
    conf = np.asarray(conf)
    order = np.argsort(-conf)
    top = min(len(conf), 200)
    c = conf[order[:top]]
    draws = rng.random((sims, top)) < c
    wrong = np.cumsum(~draws, 1)
    nn = np.arange(1, top + 1)
    risk = wrong / nn
    val = nn / len(conf) * np.clip(1 - (risk - 0.2) / 0.1, 0, 1)
    ev = val.mean(0)
    best = int(np.argmax(ev)) + 1
    keep = set(order[:best].tolist())
    return [i not in keep for i in range(len(conf))], ev[best - 1]


for seed in [0, 1, 2]:
    pl = oof(all_f["hp14"], 0.05, None, seed)
    conf = oof(make_x(all_f, pl, "plain"), 1.0, np.where(ood, 5.0, 1.0), seed)
    for nm, m in [("train", src == 0), ("dev", src == 1), ("ood", ood)]:
        c = list(conf[m])
        yy = list(y[m])
        s0 = selective(yy, pick_abstain(c, 0.0))
        ab, ev = exp_pick(c)
        s1 = selective(yy, ab)
        print("seed %d %-5s n %4d | current rule: sel %.4f (cov %.3f risk %.3f) | expected-value rule: predicted %.4f realised %.4f (cov %.3f risk %.3f)" % (
            seed, nm, len(c), s0[0], s0[1], s0[2], ev, s1[0], s1[1], s1[2]))
