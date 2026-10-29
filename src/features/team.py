"""Team identifiers: teams are reused across battles, so they identify (groups of) players."""

import hashlib
from itertools import combinations

NAN = float("nan")


def team_id(names):
    """Stable numeric id of a set of species (exact in float64)."""
    return float(int(hashlib.md5("|".join(sorted(names)).encode()).hexdigest()[:12], 16))


def team_keys(b):
    """Ids of the P1 team, of the P2 team when all six were revealed (or of the five revealed ones), and of
    the five-species subsets of both full teams, used to match a partially revealed P2 team to a known one.
    Teams are reused across battles, so they identify (groups of) players; the ids are only used through
    TeamEffect."""
    p1, rev2 = [p["name"] for p in b.team], list(b.sides["p2"].revealed)
    f = {"k_p1team": team_id(p1),
         "k_p2team": team_id(rev2) if len(rev2) == 6 else NAN,
         "k_p2part": team_id(rev2) if len(rev2) == 5 else NAN}
    subs1 = [team_id(s) for s in combinations(sorted(p1), 5)]
    subs2 = [team_id(s) for s in combinations(sorted(rev2), 5)] if len(rev2) == 6 else [NAN] * 6
    for k in range(6):
        f[f"k_p1sub{k}"], f[f"k_p2sub{k}"] = subs1[k], subs2[k]
    return f
