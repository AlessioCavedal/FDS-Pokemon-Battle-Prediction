"""Symmetric bags (Player 1 minus Player 2) of species, alive HP and revealed moves, over vocabularies
built from the training battles."""

from collections import Counter


class Vocab:
    """Species and move vocabularies built from training battles only."""

    def __init__(self, battles, min_species=20, min_moves=30):
        sp, mv = Counter(), Counter()
        for b in battles:
            sp.update(p["name"] for p in b.team)
            sp.update(b.sides["p2"].revealed.keys())
            for s in ("p1", "p2"):
                mv.update({m for moves in b.sides[s].revealed.values() for m in moves})
        self.species = sorted(k for k, v in sp.items() if v >= min_species)
        self.moves = sorted(k for k, v in mv.items() if v >= min_moves)


def bag_features(b, vocab, t1, t2):
    f = {}
    team = {p["name"] for p in b.team}
    rev2 = set(b.sides["p2"].revealed)
    for s in vocab.species:
        f[f"bs_{s}"] = float(s in team) - float(s in rev2)
        h1 = (1 - t1["pf"][s]) * t1["hp"][s] if s in team else 0.0
        h2 = (1 - t2["pf"][s]) * t2["hp"][s] if s in rev2 else 0.0
        f[f"bh_{s}"] = h1 - h2
    m1 = {m for mv in b.sides["p1"].revealed.values() for m in mv}
    m2 = {m for mv in b.sides["p2"].revealed.values() for m in mv}
    for m in vocab.moves:
        f[f"bm_{m}"] = float(m in m1) - float(m in m2)
    return f
