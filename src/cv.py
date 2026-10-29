"""Repeated stratified group K-fold cross-validation (exact duplicate timelines share a group), out-of-fold
predictions, paired fold comparisons and the experiment log."""

import csv
import json
import time

import numpy as np
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from .config import CACHE_DIR, CV_REPEATS, CV_SPLITS, EXP_DIR, SEED
from .models import fit_weighted

OOF_DIR = CACHE_DIR / "oof"
OOF_DIR.mkdir(exist_ok=True)


def make_folds(y, groups, n_splits=CV_SPLITS, n_repeats=CV_REPEATS, seed=SEED):
    y, groups = np.asarray(y), np.asarray(groups)
    return [list(StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed + r).split(y, y, groups))
            for r in range(n_repeats)]


def evaluate(oof, y):
    y = np.asarray(y)
    accs = [accuracy_score(y, p > 0.5) for p in oof]
    return {"acc": float(np.mean(accs)), "acc_rep_std": float(np.std(accs)),
            "logloss": float(np.mean([log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)) for p in oof])),
            "auc": float(np.mean([roc_auc_score(y, p) for p in oof]))}


def run(factory, X, y, folds, name=None, log=True, notes="", params=None, train_set=None):
    """train_set(X_tr, y_tr) -> (X_fit, y_fit, weights) optionally expands the training rows."""
    y = np.asarray(y)
    oof = np.zeros((len(folds), len(y)))
    fold_acc = []
    t0 = time.time()
    for r, rep in enumerate(folds):
        for tr, va in rep:
            Xt, yt, wt = train_set(X.iloc[tr], y[tr]) if train_set else (X.iloc[tr], y[tr], None)
            model = fit_weighted(factory(), Xt, yt, wt)
            oof[r, va] = model.predict_proba(X.iloc[va])[:, 1]
            fold_acc.append(accuracy_score(y[va], oof[r, va] > 0.5))
    res = evaluate(oof, y)
    res["fold_acc_std"] = float(np.std(fold_acc))
    res["seconds"] = round(time.time() - t0, 1)
    res["n_features"] = X.shape[1]
    if name:
        np.save(OOF_DIR / f"{name}.npy", oof)
        if log:
            _log(name, res, notes, params)
    return res, oof


def _log(name, res, notes, params):
    path = EXP_DIR / "log.csv"
    row = {"time": time.strftime("%Y-%m-%d %H:%M"), "name": name, **{k: round(v, 5) if isinstance(v, float) else v
                                                                      for k, v in res.items()},
           "notes": notes, "params": json.dumps(params or {}, sort_keys=True)}
    new = not path.exists()
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def paired(oof_a, oof_b, y, folds):
    """Mean and std of per-fold accuracy differences (b - a) on identical folds."""
    y = np.asarray(y)
    d = [accuracy_score(y[va], oof_b[r, va] > 0.5) - accuracy_score(y[va], oof_a[r, va] > 0.5)
         for r, rep in enumerate(folds) for _, va in rep]
    return float(np.mean(d)), float(np.std(d) / np.sqrt(len(d)))
