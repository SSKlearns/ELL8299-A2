exec(open("exp_v2cmp.py").read().split("res = {}")[0])
res = {}
keep = {}
for sd in range(10):
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=sd).split(np.zeros(n), strat))
    pl = np.zeros(n)
    for tr, te in folds:
        h = lr(0.05)
        h.fit(all_f["hp14"][tr], y[tr])
        pl[te] = h.predict_proba(all_f["hp14"][te])[:, 1]
    x1 = v1_x(pl)
    x2 = make_x(all_f, pl)
    p_v1 = np.zeros(n)
    p_two = np.zeros(n)
    p_idonly = np.zeros(n)
    for tr, te in folds:
        h = lr(1.0)
        h.fit(x1[tr], y[tr], logisticregression__sample_weight=np.where(ood[tr], 5.0, 1.0))
        p_v1[te] = h.predict_proba(x1[te])[:, 1]
        p_two[te] = two(x2, tr, te, sd, "sep")
        t2 = tr[dom[tr] == 0]
        h = lr(0.3)
        h.fit(x2[t2], y[t2])
        p_idonly[te] = h.predict_proba(x2[te])[:, 1]
    outs = {"v1": p_v1, "v2 two stage": p_two,
            "routed: ID v2 two stage, OOD v1": np.where(dom == 1, p_v1, p_two),
            "routed: ID v2 trained on ID only, OOD v1": np.where(dom == 1, p_v1, p_idonly)}
    keep[sd] = outs
    for name, p in outs.items():
        r = []
        for m in [src == 1, ood, src == 0]:
            sc, a, e, s, cov, risk = q3_score(list(p[m]), list(y[m]), pick_abstain(list(p[m])))
            r += [a, e, s, sc]
        res.setdefault(name, []).append(r)
print("10 fold seeds, EV abstention rule everywhere")
for name, v in res.items():
    o = np.mean(v, 0)
    mean_set = [(r[3] + r[7]) / 2 for r in v]
    print("%-44s dev AUROC %.4f ECE %.4f sel %.4f set %.4f | ood AUROC %.4f ECE %.4f sel %.4f set %.4f | train AUROC %.4f ECE %.4f set %.4f | MEAN(dev,ood) %.4f (sd %.4f)" % (
        name, o[0], o[1], o[2], o[3], o[4], o[5], o[6], o[7], o[8], o[9], o[11], np.mean(mean_set), np.std(mean_set)))
rng = np.random.default_rng(0)
ix = np.where(ood)[0]
d = []
for t in range(1000):
    s = rng.choice(ix, len(ix))
    a = keep[0]["v1"][s]
    b = keep[0]["v2 two stage"][s]
    ys = list(y[s])
    sa = q3_score(list(a), ys, [True] * len(s))
    sb = q3_score(list(b), ys, [True] * len(s))
    d.append(((sb[1] - sb[2]) - (sa[1] - sa[2])) / 3)
d = np.array(d)
print("OOD set score (without selective), v2 two stage minus v1, bootstrap over the 200 OOD questions: %.4f, 95%% CI [%.4f, %.4f], P(v2 better) %.2f" % (d.mean(), np.percentile(d, 2.5), np.percentile(d, 97.5), (d > 0).mean()))
