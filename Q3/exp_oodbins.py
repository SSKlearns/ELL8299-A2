import sys
sys.argv = ["x"]
exec(open("exp_v2cmp.py").read().split("res = {}")[0])
from sklearn.metrics import roc_auc_score
eces = {"v1": [], "v2": [], "v2_oodonly_v1feats": []}
bins_acc = {"v1": np.zeros((10, 3)), "v2": np.zeros((10, 3))}
for sd in range(10):
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=sd).split(np.zeros(n), strat))
    pl = np.zeros(n)
    for tr, te in folds:
        h = lr(0.05)
        h.fit(all_f["hp14"][tr], y[tr])
        pl[te] = h.predict_proba(all_f["hp14"][te])[:, 1]
    x1 = v1_x(pl)
    x2 = make_x(all_f, pl)
    p1 = np.zeros(n)
    p2 = np.zeros(n)
    for tr, te in folds:
        h = lr(1.0)
        h.fit(x1[tr], y[tr], logisticregression__sample_weight=np.where(ood[tr], 5.0, 1.0))
        p1[te] = h.predict_proba(x1[te])[:, 1]
        h = lr(0.3)
        h.fit(x2[tr], y[tr], logisticregression__sample_weight=np.where(ood[tr], 3.0, 1.0))
        p2[te] = h.predict_proba(x2[te])[:, 1]
    for nm, p in [("v1", p1), ("v2", p2)]:
        q = p[ood]
        yy = y[ood]
        sc = q3_score(list(q), list(yy), [True] * len(q))
        eces[nm].append(sc[2])
        for b in range(10):
            m = (q > b / 10) & (q <= (b + 1) / 10) if b > 0 else q <= 0.1
            bins_acc[nm][b] += [m.sum(), q[m].sum(), yy[m].sum()]
print("OOD ECE per fold seed  v1:", " ".join("%.3f" % v for v in eces["v1"]))
print("OOD ECE per fold seed  v2:", " ".join("%.3f" % v for v in eces["v2"]))
d = np.array(eces["v2"]) - np.array(eces["v1"])
print("v2 - v1: mean %.4f, v2 worse on %d / 10 seeds" % (d.mean(), (d > 0).sum()))
for nm in ["v1", "v2"]:
    print(nm, "OOD reliability (pooled over 10 seeds): bin n / mean conf / accuracy")
    for b in range(10):
        c, s, yy = bins_acc[nm][b]
        if c > 0:
            print("   bin %d  n %5.0f  conf %.3f  acc %.3f  contribution to ECE %.4f" % (b, c, s / c, yy / c, c / (200 * 10) * abs(yy / c - s / c)))
