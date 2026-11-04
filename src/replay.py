"""Timeline parsing: state of both players at each decision point of Player 1, fresh move events, Spikes
and the transitions used by the hidden-faint detector. Player 2's lead counts as revealed even when it
switches out on turn 1 and never shows up in the timeline again."""

import math

from . import mechanics as mx
from .data import effectiveness

SIDES = ("p1", "p2")
OPP = {"p1": "p2", "p2": "p1"}
STATUSES = ("slp", "par", "frz", "tox", "psn", "brn")
VOLATILE = ("substitute", "confusion", "leechseed", "perish", "trapped", "encore", "attract",
            "destinybond", "protect", "curse", "nightmare")
NAN = float("nan")


class Side:
    """Per-entry view of one player. Each timeline entry is a decision point of player 1;
    the move field holds the last move used by the active Pokemon, so fresh uses are
    detected as changes of that field (events)."""

    def __init__(self, timeline, key):
        self.names, self.hp, self.status, self.effects, self.boosts, self.moves = [], [], [], [], [], []
        for t in timeline:
            ps = t[f"{key}_pokemon_state"]
            self.names.append(ps["name"])
            self.hp.append(ps["hp_pct"])
            self.status.append(ps["status"])
            self.effects.append(frozenset(e for e in ps["effects"] if e != "noeffect"))
            self.boosts.append(ps["boosts"])
            self.moves.append(t[f"{key}_move_details"])
        self.n = len(self.names)
        self.events = self._events()
        self.revealed = {}
        for name, m in zip(self.names, self.moves):
            moves = self.revealed.setdefault(name, {})
            if m is not None:
                moves[m["name"]] = m
        self.first_index, self.last_index = {}, {}
        for i, name in enumerate(self.names):
            self.first_index.setdefault(name, i)
            self.last_index[name] = i

    def _events(self):
        events = [None] * self.n
        last = {}
        for i in range(self.n):
            name, m = self.names[i], self.moves[i]
            if m is None:
                continue
            if i > 0 and self.names[i - 1] == name:
                prev = self.moves[i - 1]
                ref = prev["name"] if prev else None
            else:
                ref = last.get(name)
            if ref != m["name"]:
                events[i] = m
            last[name] = m["name"]
        return events


class Battle:
    def __init__(self, raw, dex, cut=None):
        self.id = raw["battle_id"]
        self.team = raw["p1_team_details"]
        self.lead2 = raw["p2_lead_details"]
        self.dex = dex.for_battle(raw)
        self.levels = {"p1": {p["name"]: p["level"] for p in self.team},
                       "p2": {self.lead2["name"]: self.lead2["level"]}}
        self.items = {p["name"]: p["item"] for p in self.team}
        tl = raw["battle_timeline"][:cut]
        self.n = len(tl)
        self.sides = {s: Side(tl, s) for s in SIDES}
        self.sides["p2"].revealed.setdefault(self.lead2["name"], {})
        self.spikes = {s: self._spikes(s) for s in SIDES}
        self.p2_faint = {}

    def level(self, side, name):
        return self.levels[side].get(name, 100)

    def item(self, side, name):
        return self.items.get(name) if side == "p1" else None

    def _spikes(self, s):
        own, opp = self.sides[s], self.sides[OPP[s]]
        on, out = False, []
        for i in range(self.n):
            e = opp.events[i]
            if e is not None and e["name"] == "spikes":
                on = True
            e = own.events[i]
            if e is not None and e["name"] == "rapidspin":
                on = False
            out.append(on)
        return out

    def transitions(self, s):
        """Active-Pokemon changes of side s. For p1, faints are recorded explicitly; the
        faint entry is skipped so that p1 faints look like the hidden faints of p2."""
        own, opp = self.sides[s], self.sides[OPP[s]]
        rows = []
        last_hp, last_mv, seen = {}, {}, set()
        for i in range(1, self.n):
            j = i - 1
            name = own.names[j]
            last_hp[name] = own.hp[j]
            seen.add(name)
            if own.moves[j] is not None:
                last_mv[name] = own.moves[j]["name"]
            x, y = own.names[j], own.names[i]
            if x == y:
                continue
            final = own.last_index[x] == j
            if own.status[j] == "fnt":
                usable = s == "p1" and opp.status[j] != "fnt" and i >= 2 and own.names[i - 2] == x
                if not usable:
                    rows.append({"i": i, "x": x, "recorded": True, "final": True, "label": 1})
                    continue
                pre, label, recorded = i - 2, 1, True
            else:
                pre, label, recorded = i - 1, 0, False
            row = {"i": i, "x": x, "recorded": recorded, "final": final, "label": label}
            row.update(self._transition_features(s, pre, i, x, y, last_hp, last_mv, seen))
            rows.append(row)
        return rows

    def _transition_features(self, s, pre, i, x, y, last_hp, last_mv, seen):
        own, opp = self.sides[s], self.sides[OPP[s]]
        o = OPP[s]
        xd, yd = self.dex.get(x), self.dex.get(y)
        a_pre, a_arr = opp.names[pre], opp.names[i]
        ad = self.dex.get(a_arr)
        am = opp.moves[i]
        a_same = a_pre == a_arr
        dmg = mx.damage_fraction(am, ad, xd, att_level=self.level(o, a_arr), def_level=self.level(s, x),
                                 att_boosts=opp.boosts[i], def_boosts=own.boosts[pre],
                                 att_item=self.item(o, a_arr), att_status=opp.status[i], att_name=a_arr)
        st, eff = own.status[pre], own.effects[pre]
        y_prev = last_hp.get(y, 1.0)
        ym = own.moves[i]
        prev_am = opp.moves[pre]
        return {
            "turn": i,
            "x_hp": own.hp[pre],
            "x_slp": st == "slp", "x_par": st == "par", "x_frz": st == "frz",
            "x_psn": st in ("psn", "tox"), "x_brn": st == "brn",
            "x_boost": sum(own.boosts[pre].values()),
            "x_trapped": "trapped" in eff, "x_perish": "perish" in eff, "x_sub": "substitute" in eff,
            "x_seed": "leechseed" in eff, "x_curse": "curse" in eff, "x_conf": "confusion" in eff,
            "x_dbond": "destinybond" in eff,
            "x_bhp": xd["hp"] if xd else NAN, "x_bdef": xd["def"] if xd else NAN,
            "x_bspd": xd["spd"] if xd else NAN,
            "y_first": y not in seen, "y_hp": own.hp[i], "y_prev_hp": y_prev, "y_drop": y_prev - own.hp[i],
            "y_mv_none": ym is None, "y_mv_new": ym is not None and last_mv.get(y) != ym["name"],
            "y_bst": yd["bst"] if yd else NAN,
            "spikes": self.spikes[s][i - 1],
            "a_same": a_same,
            "a_drop": opp.hp[pre] - opp.hp[i] if a_same else NAN,
            "a_mv_changed": am is not None and (not a_same or prev_am is None or prev_am["name"] != am["name"]),
            "a_bp": am["base_power"] if am else 0,
            "a_status_mv": am is not None and am["category"] == "status",
            "a_eff": effectiveness(am["type"], xd["types"]) if (am and xd) else NAN,
            "a_dmg": dmg,
            "x_hp_minus_dmg": own.hp[pre] - dmg,
            "a_phaze": am is not None and am["name"] in mx.PHAZE,
            "a_boost": sum(opp.boosts[i].values()),
            "a_disabled": opp.status[i] in ("slp", "frz"),
        }


def parse(raws, dex, cut=None):
    return [Battle(r, dex, cut) for r in raws]


def safe_div(a, b):
    return a / b if b else 0.0


def log_ratio(a, b):
    if not (a > 0 and b > 0) or math.isnan(a) or math.isnan(b):
        return NAN
    return math.log(a / b)
