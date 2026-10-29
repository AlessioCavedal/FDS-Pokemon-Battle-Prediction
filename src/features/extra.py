"""Sleep-related features, plus time-share and count variants of the species and move bags (an experiment
not used by the final pipelines)."""

import math
from collections import Counter


def sleep_features(b, t1, t2):
    f = {}
    for key, track in (("p1", t1), ("p2", t2)):
        side = b.sides[key]
        prev, spell = {}, {}
        for i in range(b.n):
            n, st = side.names[i], side.status[i]
            if st == "slp":
                if prev.get(n) != "slp" or spell.get(n) is None:
                    m = side.moves[i]
                    spell[n] = [0, m is not None and m["name"] == "rest"]
                spell[n][0] += 1
            else:
                spell[n] = None
            prev[n] = st
        sleepers = [n for n, sp in spell.items() if sp is not None and track["pf"].get(n, 0) < 0.5]
        f[f"x_{key}_sleep_turns_max"] = max((spell[n][0] for n in sleepers), default=0)
        f[f"x_{key}_rest_sleepers"] = sum(spell[n][1] for n in sleepers)
        f[f"x_{key}_induced_sleepers"] = sum(not spell[n][1] for n in sleepers)
    for k in ("sleep_turns_max", "rest_sleepers", "induced_sleepers"):
        f[f"x_d_{k}"] = f[f"x_p1_{k}"] - f[f"x_p2_{k}"]
    return f


def usage_bags(b, vocab):
    f = {}
    t1, t2 = Counter(b.sides["p1"].names), Counter(b.sides["p2"].names)
    for s in vocab.species:
        f[f"bt_{s}"] = (t1.get(s, 0) - t2.get(s, 0)) / b.n
    c1 = Counter(m["name"] for m in b.sides["p1"].events if m is not None)
    c2 = Counter(m["name"] for m in b.sides["p2"].events if m is not None)
    for m in vocab.moves:
        f[f"bc_{m}"] = math.log1p(c1.get(m, 0)) - math.log1p(c2.get(m, 0))
    return f
