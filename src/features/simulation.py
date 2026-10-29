"""Duel between the two actives and a quick simulation of the rest of the battle as a sequence of
one-on-one duels, with the HP carried over and the best answer sent in after each knockout. First version:
revealed plus typical moves of each species, or revealed moves only. Second version (simulation_features):
typical moves and a STAB move per type always available, turns still lost asleep or frozen, weather and
screens in the duel of the actives, and a variant against the unrevealed Player 2 Pokémon guessed from team
co-occurrences."""

import math
from collections import Counter

from .. import mechanics as mx
from .matchup import DISABLE, _proxies

MAX_HITS = 30
MAX_DUELS = 15
MIN_HP = 0.01
TYPICAL_MOVES, MIN_COUNT, MIN_SHARE = 4, 3, 0.04
FIELD_TURNS = 5
WEATHER_BOOST = {"raindance": {"water": 1.5, "fire": 0.5}, "sunnyday": {"fire": 1.5, "water": 0.5}}
SCREEN_OF = {"physical": "reflect", "special": "lightscreen"}


def _attack(m):
    return (m["category"] != "status" and m["name"] not in mx.SELF_KO
            and (m["base_power"] > 1 or m["name"] in mx.FIXED_DAMAGE))


class MoveTable:
    """Typical attacking moves of each species (the most often revealed ones in the training battles) and the
    typical length, in entries spent active, of sleep after Rest, other sleep and freeze."""

    def __init__(self, battles):
        counts, known = {}, {}
        for b in battles:
            for side in b.sides.values():
                for name, moves in side.revealed.items():
                    for m in moves.values():
                        if _attack(m):
                            counts.setdefault(name, Counter())[m["name"]] += 1
                            known[m["name"]] = m
        self.typical = {
            name: [known[m] for m, n in c.most_common(TYPICAL_MOVES)
                   if n >= MIN_COUNT and n / sum(c.values()) >= MIN_SHARE]
            for name, c in counts.items()}
        self.turns = _spell_lengths(battles)


def _spell_lengths(battles):
    lengths = {"rest": [], "slp": [], "frz": []}
    for b in battles:
        for side in b.sides.values():
            run, kind = 0, None
            for i in range(b.n):
                st, same = side.status[i], i > 0 and side.names[i - 1] == side.names[i]
                if st in ("slp", "frz") and same and side.status[i - 1] == st:
                    run += 1
                elif st in ("slp", "frz"):
                    rest = st == "slp" and side.moves[i] is not None and side.moves[i]["name"] == "rest"
                    run, kind = 1, "rest" if rest else st
                else:
                    if run and same and side.status[i - 1] in ("slp", "frz"):
                        lengths[kind].append(run)
                    run, kind = 0, None
    return {k: round(sum(v) / len(v)) if v else 2 for k, v in lengths.items()}


def _fighters(b, side_key, track, alive, table, active):
    side, i = b.sides[side_key], b.n - 1
    out = []
    for n, hp in track["hp"].items():
        d = b.dex.get(n)
        if d is None or hp <= 0 or not alive(n):
            continue
        moves = {m["name"]: m for m in side.revealed.get(n, {}).values() if _attack(m)}
        for m in table.typical.get(n, []) if table else []:
            moves.setdefault(m["name"], m)
        st = track["status"][n]
        boosts = side.boosts[i] if n == active else {}
        level = b.level(side_key, n)
        mult = 1.0
        if st in DISABLE and not (st == "slp" and "sleeptalk" in side.revealed.get(n, {})):
            mult = DISABLE[st]
        out.append({"name": n, "hp": hp, "d": d, "level": level, "item": b.item(side_key, n), "status": st,
                    "boosts": boosts, "mult": mult, "moves": list(moves.values()) or _proxies(d["types"]),
                    "spe": mx.effective_speed(d, boosts, st, level)})
    return out


def _damage(u, v, cache):
    key = (u["name"], v["name"])
    if key not in cache:
        cache[key] = u["mult"] * mx.best_damage(u["moves"], u["d"], v["d"], att_level=u["level"], def_level=v["level"],
                                                att_boosts=u["boosts"], def_boosts=v["boosts"], att_item=u["item"],
                                                att_status=u["status"], att_name=u["name"])
    return cache[key]


def _hits(u, v, cache):
    dmg = _damage(u, v, cache)
    return min(MAX_HITS, math.ceil(v["hp"] / dmg)) if dmg > 0 else MAX_HITS


def _duel(a, b, cache):
    """(a wins, HP left to the winner)."""
    ha, hb = _hits(a, b, cache), _hits(b, a, cache)
    a_fast = a["spe"] > b["spe"]
    if ha < hb or (ha == hb and a_fast):
        return True, max(MIN_HP, a["hp"] - (ha - 1 if a_fast else ha) * _damage(b, a, cache))
    return False, max(MIN_HP, b["hp"] - (hb if a_fast else hb - 1) * _damage(a, b, cache))


def _best_answer(team, foe, cache):
    return max(range(len(team)), key=lambda k: math.log(_hits(foe, team[k], cache) / _hits(team[k], foe, cache)))


def simulate(A, B, act_a, act_b):
    A, B, cache = list(A), list(B), {}
    ia = next((k for k, u in enumerate(A) if u["name"] == act_a), None)
    ib = next((k for k, v in enumerate(B) if v["name"] == act_b), None)
    if ia is None and A and B:
        ia = _best_answer(A, B[ib if ib is not None else 0], cache)
    if ib is None and A and B:
        ib = _best_answer(B, A[ia], cache)
    for _ in range(MAX_DUELS):
        if not A or not B:
            break
        a_wins, hp = _duel(A[ia], B[ib], cache)
        if a_wins:
            A[ia] = {**A[ia], "hp": hp}
            B.pop(ib)
            if B:
                ib = _best_answer(B, A[ia], cache)
        else:
            B[ib] = {**B[ib], "hp": hp}
            A.pop(ia)
            if A:
                ia = _best_answer(A, B[ib], cache)
    return A, B


def _outcomes(b, t1, t2, table, suffix):
    i = b.n - 1
    n1, n2 = b.sides["p1"].names[i], b.sides["p2"].names[i]
    A = _fighters(b, "p1", t1, lambda n: t1["pf"][n] < 0.5, table, n1)
    B = _fighters(b, "p2", t2, lambda n: t2["pf"][n] < 0.5, table, n2)
    f = {}
    a = next((u for u in A if u["name"] == n1), None)
    v = next((u for u in B if u["name"] == n2), None)
    if a is None or v is None:
        f["d_act_duel" + suffix] = 0.0
    else:
        cache = {}
        ha, hb = _hits(a, v, cache), _hits(v, a, cache)
        if ha == hb:
            ha, hb = v["spe"], a["spe"]
        f["d_act_duel" + suffix] = float(ha < hb) - float(ha > hb)
    left_a, left_b = simulate(A, B, n1, n2)
    f["d_sim_left" + suffix] = (len(left_a) - len(left_b)) / 6
    f["d_sim_hp" + suffix] = (sum(u["hp"] for u in left_a) - sum(u["hp"] for u in left_b)) / 6
    return f


def duel_features(b, t1, t2, table):
    return {**_outcomes(b, t1, t2, table, ""), **_outcomes(b, t1, t2, None, "_seen")}


def used_move(b, s, i):
    """Move used at entry i by the active Pokémon of side s: the move field also counts when the same move is
    used again, but not on a switch-in, while it cannot act, or right after a Player 1 faint."""
    side = b.sides[s]
    if i == 0:
        return side.moves[0]
    if b.sides["p1"].status[i - 1] == "fnt" or side.names[i] != side.names[i - 1]:
        return None
    if side.status[i - 1] in ("slp", "frz") and side.status[i] in ("slp", "frz"):
        return None
    return side.moves[i]


def _field(b):
    """Weather and screens still up at the last entry."""
    weather, screens = (None, -FIELD_TURNS), {"p1": {}, "p2": {}}
    for i in range(b.n):
        for s in ("p1", "p2"):
            m = used_move(b, s, i)
            if m is not None and m["name"] in mx.WEATHER:
                weather = (m["name"], i)
            elif m is not None and m["name"] in mx.SCREENS:
                screens[s][m["name"]] = i
    end = b.n - 1
    return (weather[0] if end - weather[1] < FIELD_TURNS else None,
            {s: {k for k, j in v.items() if end - j < FIELD_TURNS} for s, v in screens.items()})


def _spent(b, s):
    """Entries each Pokémon has spent in its current sleep or freeze, and the kind of spell."""
    side, spent, kind, prev = b.sides[s], {}, {}, {}
    for i in range(b.n):
        n, st = side.names[i], side.status[i]
        if st in ("slp", "frz"):
            if prev.get(n) == st:
                spent[n] += 1
            else:
                spent[n] = 1
                rest = st == "slp" and side.moves[i] is not None and side.moves[i]["name"] == "rest"
                kind[n] = "rest" if rest else st
        prev[n] = st
    return spent, kind


def _fighter(n, side_key, hp, d, level, item, status, boosts, revealed, table, delay):
    moves = {m["name"]: m for m in revealed.values() if _attack(m)}
    for m in table.typical.get(n, []) + _proxies(d["types"]):
        moves.setdefault(m["name"], m)
    return {"name": n, "side": side_key, "hp": hp, "d": d, "level": level, "item": item, "status": status,
            "boosts": boosts, "moves": list(moves.values()), "delay": delay,
            "spe": mx.effective_speed(d, boosts, status, level)}


def _fighters2(b, side_key, track, alive, table, active):
    side, i = b.sides[side_key], b.n - 1
    spent, kind = _spent(b, side_key)
    out = []
    for n, hp in track["hp"].items():
        d = b.dex.get(n)
        if d is None or hp <= 0 or not alive(n):
            continue
        revealed, st = side.revealed.get(n, {}), track["status"][n]
        delay = 0
        if st == "slp" and "sleeptalk" not in revealed:
            delay = max(0, table.turns[kind.get(n, "slp")] - spent.get(n, 0))
        elif st == "frz":
            delay = max(0, table.turns["frz"] - spent.get(n, 0))
        out.append(_fighter(n, side_key, hp, d, b.level(side_key, n), b.item(side_key, n), st,
                            side.boosts[i] if n == active else {}, revealed, table, delay))
    return out


def _hidden(b, table, team_model):
    """Most likely unrevealed Player 2 Pokémon, at full HP."""
    revealed = list(b.sides["p2"].revealed)
    n_hidden = 6 - len(revealed)
    if n_hidden <= 0:
        return []
    e = team_model.expected_unrevealed(revealed, n_hidden)
    names = [n for n, _ in sorted(e.items(), key=lambda kv: -kv[1]) if n in b.dex][:n_hidden]
    return [_fighter(n, "p2", 1.0, b.dex[n], 100, None, "nostatus", {}, {}, table, 0) for n in names]


def _damage2(u, v, field):
    best = 0.0
    for m in u["moves"]:
        x = mx.damage_fraction(m, u["d"], v["d"], att_level=u["level"], def_level=v["level"], att_boosts=u["boosts"],
                               def_boosts=v["boosts"], att_item=u["item"], att_status=u["status"], att_name=u["name"])
        if field is not None:
            weather, screens = field
            x *= WEATHER_BOOST.get(weather, {}).get(m["type"], 1.0)
            if SCREEN_OF.get(m["category"]) in screens[v["side"]]:
                x *= 0.5
        best = max(best, x)
    return best


def _need(u, v, cache, field=None):
    """Turns u needs to knock v out: hits plus the turns it still loses asleep or frozen."""
    key = (u["name"], v["name"])
    if key not in cache:
        cache[key] = _damage2(u, v, field)
    dmg = cache[key]
    return min(MAX_HITS, math.ceil(v["hp"] / dmg) + u["delay"]) if dmg > 0 else MAX_HITS


def _duel2(a, b, cache):
    ta, tb = _need(a, b, cache), _need(b, a, cache)
    a_fast = a["spe"] > b["spe"]
    if ta < tb or (ta == tb and a_fast):
        hits = max(0, (ta - 1 if a_fast else ta) - b["delay"])
        return True, max(MIN_HP, a["hp"] - hits * cache[b["name"], a["name"]])
    hits = max(0, (tb if a_fast else tb - 1) - a["delay"])
    return False, max(MIN_HP, b["hp"] - hits * cache[a["name"], b["name"]])


def _answer2(team, foe, cache):
    return max(range(len(team)), key=lambda k: (math.log(_need(foe, team[k], cache) / _need(team[k], foe, cache)),
                                                team[k]["spe"]))


def simulate2(A, B, act_a, act_b):
    A, B, cache = list(A), list(B), {}
    ia = next((k for k, u in enumerate(A) if u["name"] == act_a), None)
    ib = next((k for k, v in enumerate(B) if v["name"] == act_b), None)
    if ia is None and A and B:
        ia = _answer2(A, B[ib if ib is not None else 0], cache)
    if ib is None and A and B:
        ib = _answer2(B, A[ia], cache)
    for _ in range(MAX_DUELS):
        if not A or not B:
            break
        a_wins, hp = _duel2(A[ia], B[ib], cache)
        if a_wins:
            A[ia] = {**A[ia], "hp": hp, "delay": 0}
            B.pop(ib)
            if B:
                ib = _answer2(B, A[ia], cache)
        else:
            B[ib] = {**B[ib], "hp": hp, "delay": 0}
            A.pop(ia)
            if A:
                ia = _answer2(A, B[ib], cache)
    return A, B


def simulation_features(b, t1, t2, table, team_model):
    i = b.n - 1
    n1, n2 = b.sides["p1"].names[i], b.sides["p2"].names[i]
    A = _fighters2(b, "p1", t1, lambda n: t1["pf"][n] < 0.5, table, n1)
    B = _fighters2(b, "p2", t2, lambda n: t2["pf"][n] < 0.5, table, n2)
    a = next((u for u in A if u["name"] == n1), None)
    v = next((u for u in B if u["name"] == n2), None)
    duel = 0.0
    if a is not None and v is not None:
        field, cache = _field(b), {}
        ta, tv = _need(a, v, cache, field), _need(v, a, cache, field)
        if ta == tv:
            ta, tv = v["spe"], a["spe"]
        duel = float(ta < tv) - float(ta > tv)
    f = {"z_act_duel": duel}
    for tag, foes in (("seen", B), ("full", B + _hidden(b, table, team_model))):
        left_a, left_b = simulate2(A, foes, n1, n2)
        f[f"z_sim_{tag}_left"] = (len(left_a) - len(left_b)) / 6
        f[f"z_sim_{tag}_hp"] = (sum(u["hp"] for u in left_a) - sum(u["hp"] for u in left_b)) / 6
    return f
