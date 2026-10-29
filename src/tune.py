"""Hyper-parameter search with Optuna on precomputed cross-fitted stacking features
(`python -m src.tune <pipeline> --trials N`); the best parameters are stored in configs/tuned.json."""

import argparse
import json

import numpy as np
import optuna
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline

from . import cv, models, pipelines
from .config import CACHE_DIR, CONFIG_DIR, SEED
from .features.build import load

TUNE_SEED = SEED + 100


def space(trial, model):
    if model == "lgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 3000, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 0.003, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 4, 64, log=True),
            "max_depth": trial.suggest_categorical("max_depth", [-1, 3, 4, 5, 6, 8]),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 300, log=True),
            "subsample": trial.suggest_float("subsample", 0.4, 1.0),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.05, 0.8),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 100, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.5),
        }
    if model == "xgb":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 3000, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 0.003, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 50, log=True),
            "subsample": trial.suggest_float("subsample", 0.4, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.05, 0.8),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 100, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 2.0),
        }
    if model == "cat":
        return {
            "iterations": trial.suggest_int("iterations", 300, 3000, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
            "depth": trial.suggest_int("depth", 3, 8),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.5, 50, log=True),
            "random_strength": trial.suggest_float("random_strength", 0.1, 10, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 2.0),
            "rsm": trial.suggest_float("rsm", 0.05, 0.8),
        }
    if model == "lr":
        return {"C": trial.suggest_float("C", 1e-4, 1.0, log=True)}
    raise ValueError(model)


def prepare(spec, X, y, folds):
    cols = pipelines.columns(spec, X.columns)
    out = []
    for tr, va in folds:
        Xtr, Xva = X.iloc[tr][cols], X.iloc[va][cols]
        steps = pipelines.stack_steps(spec, cols) + pipelines.team_steps(spec, cols)
        if steps:
            pre = Pipeline(steps)
            Xtr, Xva = pre.fit_transform(Xtr, y[tr]), pre.transform(Xva)
        out.append((Xtr, y[tr], Xva, y[va]))
    return out


def run(pipeline, trials):
    d = load()
    X, y, g = d["X_train"], d["y"].to_numpy(), d["groups"].to_numpy()
    spec = pipelines.specs()[pipeline]
    folds = cv.make_folds(y, g, n_repeats=1, seed=TUNE_SEED)[0]
    data = prepare(spec, X, y, folds)

    def objective(trial):
        params = space(trial, spec["model"])
        lls, accs = [], []
        for Xtr, ytr, Xva, yva in data:
            m = models.make(spec["model"], params).fit(Xtr, ytr)
            p = m.predict_proba(Xva)[:, 1]
            lls.append(log_loss(yva, p))
            accs.append(accuracy_score(yva, p > 0.5))
        trial.set_user_attr("acc", float(np.mean(accs)))
        return float(np.mean(lls))

    storage = f"sqlite:///{CACHE_DIR / 'optuna.db'}"
    study = optuna.create_study(study_name=pipeline, storage=storage, direction="minimize", load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=SEED, multivariate=True))
    if not study.trials and spec.get("params"):
        study.enqueue_trial({k: v for k, v in spec["params"].items() if k != "subsample_freq"})
    study.optimize(objective, n_trials=trials)
    best = study.best_trial
    print(f"best ll {best.value:.5f} acc {best.user_attrs['acc']:.4f} params {best.params}")
    path = CONFIG_DIR / "tuned.json"
    tuned = json.loads(path.read_text()) if path.exists() else {}
    tuned[pipeline] = {"params": best.params, "logloss": best.value, "acc": best.user_attrs["acc"]}
    path.write_text(json.dumps(tuned, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(prog="python -m src.tune")
    ap.add_argument("pipeline")
    ap.add_argument("--trials", type=int, default=100)
    a = ap.parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    run(a.pipeline, a.trials)
