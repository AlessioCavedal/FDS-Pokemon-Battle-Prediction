"""Feature groups (selected by column prefix) and the cached feature build for train and test."""

import json
import pickle
import time

import pandas as pd

from ..config import CACHE_DIR
from ..data import load_test, load_train, pokedex
from ..faint import fit_detector
from ..replay import parse
from .active import active_features
from .bags import Vocab, bag_features
from .events import event_features
from .extra import sleep_features, usage_bags
from .hidden import TeamModel, hidden_features
from .matchup import matchup_features, wincon_features
from .state import p1_track, p2_track, state_features
from .static import static_features
from .team import team_keys

VERSION = "v6"
GROUPS = {
    "static": ("s_",), "state": ("f_",), "traj": ("t_",), "active": ("a_",), "events": ("e_",),
    "matchup": ("m_",), "wincon": ("w_",), "sleep": ("x_",),
    "bag_species": ("bs_",), "bag_hp": ("bh_",), "bag_moves": ("bm_",), "bag_time": ("bt_",), "bag_count": ("bc_",),
    "bag_full": ("bf_",), "bag_hp_full": ("bg_",), "hidden": ("u_",), "team": ("k_",), "cut": ("cut",),
}


def battle_features(b, vocab, team_model):
    t1, t2 = p1_track(b), p2_track(b)
    f = {"battle_id": b.id}
    f.update(static_features(b))
    f.update(state_features(b, t1, t2))
    f.update(active_features(b))
    f.update(event_features(b))
    f.update(matchup_features(b, t1, t2))
    f.update(wincon_features(b, t1, t2))
    f.update(sleep_features(b, t1, t2))
    f.update(bag_features(b, vocab, t1, t2))
    f.update(usage_bags(b, vocab))
    f.update(hidden_features(b, team_model, vocab, t1, t2))
    f.update(team_keys(b))
    return f


def build():
    t0 = time.time()
    train_raw, test_raw = load_train(), load_test()
    dex = pokedex()
    train, test = parse(train_raw, dex), parse(test_raw, dex)
    info = fit_detector(train, test)
    vocab, team_model = Vocab(train), TeamModel(train)
    X_train = pd.DataFrame([battle_features(b, vocab, team_model) for b in train]).set_index("battle_id")
    X_test = pd.DataFrame([battle_features(b, vocab, team_model) for b in test]).set_index("battle_id")
    X_train = X_train.astype(float)
    X_test = X_test[X_train.columns].astype(float)
    X_train["cut"] = X_test["cut"] = float(len(train_raw[0]["battle_timeline"]))
    y = pd.Series([r["player_won"] for r in train_raw], index=X_train.index, name="player_won").astype(int)
    groups = _duplicate_groups(train_raw)
    info["seconds"] = round(time.time() - t0, 1)
    return {"X_train": X_train, "X_test": X_test, "y": y, "groups": groups, "info": info}


def _duplicate_groups(raws):
    seen, groups = {}, []
    for r in raws:
        key = json.dumps(r["battle_timeline"], sort_keys=True)
        groups.append(seen.setdefault(key, len(seen)))
    return pd.Series(groups, index=[r["battle_id"] for r in raws])


def load(rebuild=False):
    path = CACHE_DIR / f"features_{VERSION}.pkl"
    if path.exists() and not rebuild:
        with open(path, "rb") as fh:
            return pickle.load(fh)
    data = build()
    with open(path, "wb") as fh:
        pickle.dump(data, fh)
    return data


def build_truncated(cuts):
    """Training features recomputed on timelines truncated to `cut` entries (same final label)."""
    train_raw = load_train()
    dex = pokedex()
    full = parse(train_raw, dex)
    vocab, team_model = Vocab(full), TeamModel(full)
    frames = []
    for cut in cuts:
        battles = parse(train_raw, dex, cut=cut)
        fit_detector(battles, [])
        X = pd.DataFrame([battle_features(b, vocab, team_model) for b in battles]).set_index("battle_id").astype(float)
        X["cut"] = float(cut)
        frames.append(X)
    return pd.concat(frames)


def load_truncated(cuts=(20, 25), rebuild=False):
    path = CACHE_DIR / f"features_{VERSION}_cuts_{'_'.join(map(str, cuts))}.pkl"
    if path.exists() and not rebuild:
        return pd.read_pickle(path)
    X = build_truncated(cuts)
    X.to_pickle(path)
    return X


def select(columns, groups):
    prefixes = tuple(p for g in groups for p in GROUPS[g])
    return [c for c in columns if c.startswith(prefixes)]
