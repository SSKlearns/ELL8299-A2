import sys
import collections
import numpy as np
from common import read


def grams(t, n):
    return [tuple(t[i:i + n]) for i in range(len(t) - n + 1)]


for path in sys.argv[1:]:
    d2 = []
    d3 = []
    rep4 = []
    loops = 0
    rows = read(path)
    for r in rows:
        t = r["token_ids"]
        g2 = grams(t, 2)
        g3 = grams(t, 3)
        g4 = grams(t, 4)
        d2.append(len(set(g2)) / len(g2))
        d3.append(len(set(g3)) / len(g3))
        c = collections.Counter(g4)
        rep4.append(sum(v - 1 for v in c.values()) / len(g4))
        if max(c.values()) >= 3:
            loops += 1
    print("%-70s distinct-2 %.3f  distinct-3 %.3f  repeated 4-grams %.3f  texts with a 4-gram x3+ %d/%d" % (
        path[-70:], np.mean(d2), np.mean(d3), np.mean(rep4), loops, len(rows)))
