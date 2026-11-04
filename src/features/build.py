"""Feature groups (selected by column prefix) and the cached feature build for train and test, including
the learned forecast fitted on the truncated training timelines."""

import json
import pickle
import time

import pandas as pd

from ..config import CACHE_DIR
from ..data import load_test, load_train, pokedex
from ..faint import fit_detector
from ..replay import parse
from . import forecast
from .active import active_features
from .bags import Vocab, bag_features
from .condition import condition_features
from .events import event_features
from .extra import sleep_features, usage_bags
from .hidden import TeamModel, hidden_features
from .matchup import matchup_features, wincon_features
from .simulation import MoveTable, duel_features, simulation_features
from .state import p1_track, p2_track, state_features
from .static import static_features
from .switching import SwitchRates, behaviour_features
from .team import team_keys

VERSION = "v10"
GROUPS = {
    "static": ("s_",), "state": ("f_",), "traj": ("t_",), "active": ("a_",), "events": ("e_",),
    "matchup": ("m_",), "wincon": ("w_",), "sleep": ("x_",),
    "bag_species": ("bs_",), "bag_hp": ("bh_",), "bag_moves": ("bm_",), "bag_time": ("bt_",), "bag_count": ("bc_",),
    "bag_full": ("bf_",), "bag_hp_full": ("bg_",), "hidden": ("u_",), "team": ("k_",), "sim": ("d_",),
    "behaviour": ("v_",), "sim2": ("z_",), "context": ("c_",), "forecast": ("q_",), "cut": ("cut",),
}


def battle_features(b, vocab, team_model, moves, rates):
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
    f.update(duel_features(b, t1, t2, moves))
    f.update(behaviour_features(b, rates))
    f.update(simulation_features(b, t1, t2, moves, team_model))
    f.update(condition_features(b, t1, t2, team_model))
    return f


def build():
    t0 = time.time()
    train_raw, test_raw = load_train(), load_test()
    dex = pokedex()
    train, test = parse(train_raw, dex), parse(test_raw, dex)
    info = fit_detector(train, test)
    helpers = Vocab(train), TeamModel(train), MoveTable(train), SwitchRates(train)
    X_train = pd.DataFrame([battle_features(b, *helpers) for b in train]).set_index("battle_id")
    X_test = pd.DataFrame([battle_features(b, *helpers) for b in test]).set_index("battle_id")
    X_train = X_train.astype(float)
    X_test = X_test[X_train.columns].astype(float)
    X_train["cut"] = X_test["cut"] = float(len(train_raw[0]["battle_timeline"]))
    truncated = truncated_features(train_raw, dex, helpers, forecast.CUTS, forecast.inputs(X_train.columns))
    Q_train, Q_test = forecast.forecast_features(train, X_train, X_test, truncated)
    X_train, X_test = X_train.join(Q_train), X_test.join(Q_test)
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


def truncated_features(train_raw, dex, helpers, cuts, columns=None):
    """Training features recomputed on timelines truncated to `cut` entries (column `cut`), with the helpers
    fitted on the full training battles and the hidden-faint detector refitted on the truncated ones."""
    frames = []
    for cut in cuts:
        battles = parse(train_raw, dex, cut=cut)
        fit_detector(battles, [])
        X = pd.DataFrame([battle_features(b, *helpers) for b in battles]).set_index("battle_id").astype(float)
        X["cut"] = float(cut)
        frames.append(X if columns is None else X[columns])
    return pd.concat(frames)


def build_truncated(cuts):
    """Training features on truncated timelines (same final label), for the augmentation experiment."""
    train_raw = load_train()
    dex = pokedex()
    full = parse(train_raw, dex)
    fit_detector(full, [])
    helpers = Vocab(full), TeamModel(full), MoveTable(full), SwitchRates(full)
    return truncated_features(train_raw, dex, helpers, cuts)


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
