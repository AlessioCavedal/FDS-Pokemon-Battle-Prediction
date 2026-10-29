"""Data loading, the supplied Gen 2 type chart and a Pokédex built from the training battles."""

import json
from functools import cache

import pandas as pd

from .config import DATA_DIR

STATS = ("hp", "atk", "def", "spa", "spd", "spe")


def load_jsonl(name):
    with open(DATA_DIR / name) as f:
        return [json.loads(line) for line in f]


@cache
def load_train():
    return load_jsonl("train.jsonl")


@cache
def load_test():
    return load_jsonl("test.jsonl")


def load_sample_submission():
    return pd.read_csv(DATA_DIR / "sample_submission.csv")


@cache
def type_chart():
    df = pd.read_csv(DATA_DIR / "type_chart_gen2.csv")
    return {(a, d): m for a, d, m in df.itertuples(index=False)}


def effectiveness(move_type, def_types):
    chart = type_chart()
    mult = 1.0
    for t in def_types:
        if t != "notype":
            mult *= chart.get((move_type, t), 1.0)
    return mult


ALL_TYPES = (
    "normal", "fire", "water", "electric", "grass", "ice", "fighting", "poison", "ground",
    "flying", "psychic", "bug", "rock", "ghost", "dragon", "dark", "steel",
)


class Pokedex:
    def __init__(self, battles):
        self.entries = {}
        for b in battles:
            for p in b["p1_team_details"] + [b["p2_lead_details"]]:
                self.entries.setdefault(p["name"], self._entry(p))

    @staticmethod
    def _entry(p):
        e = {s: p[f"base_{s}"] for s in STATS}
        e["types"] = tuple(t for t in p["types"] if t != "notype")
        e["bst"] = sum(e[s] for s in STATS)
        return e

    def for_battle(self, battle):
        local = dict(self.entries)
        for p in battle["p1_team_details"] + [battle["p2_lead_details"]]:
            if p["name"] not in local:
                local[p["name"]] = self._entry(p)
        return local


@cache
def pokedex():
    return Pokedex(load_train())
