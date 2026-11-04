"""Named pipelines from configs/pipelines.json: feature groups, stacking and team-effect steps, model
parameters and optional training-set augmentation."""

import json

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from . import models
from .config import CONFIG_DIR, SEED
from .features.build import load_truncated, select


def specs():
    with open(CONFIG_DIR / "pipelines.json") as fh:
        return json.load(fh)


def columns(spec, all_columns):
    return select(all_columns, spec["groups"] + (["team"] if spec.get("team") else []))


def stack_steps(spec, cols, seed=SEED):
    stack = spec.get("stack")
    if not stack:
        return []
    blocks = [select(cols, [g]) for g in stack] if spec.get("stack_separate") else [select(cols, stack)]
    return models.stack_steps(blocks, C=spec.get("stack_C", 0.01), seed=seed, full_C=spec.get("stack_full_C"),
                              full_cols=_no_keys(cols), full_spline_keys=spec.get("stack_full_spline"))


def _no_keys(cols):
    return [c for c in cols if not c.startswith("k_")]


def team_steps(spec, cols, seed=SEED):
    """Team effect on top of the full stacked logit; models without one get their own cross-fitted base."""
    team = spec.get("team")
    if not team:
        return []
    steps, own_base = [], not spec.get("stack_full_C")
    if own_base:
        steps.append(("team_base", models.StackedLogit(_no_keys(cols), C=team.get("base_C", 0.003), seed=seed,
                                                       name="stack_logit_full", keep=True,
                                                       spline_keys=team.get("base_spline"))))
    steps.append(("team", models.TeamEffect(a=team.get("a", 5.0), p2=team.get("p2", True),
                                            link=team.get("link", False), pair=team.get("pair", False),
                                            split=team.get("split", False), joint=team.get("joint", False),
                                            lam=team.get("lam", 1.0), lam_pair=team.get("lam_pair"), seed=seed,
                                            drop_base=own_base, soft=team.get("soft", False),
                                            soft_min=team.get("soft_min", 3))))
    return steps


def estimator(spec, cols, seed=SEED):
    est = models.make(spec["model"], spec.get("params"), seed=seed)
    if spec.get("monotone"):
        est = models.Monotone(est)
    steps = stack_steps(spec, cols, seed) + team_steps(spec, cols, seed)
    return Pipeline(steps + [("model", est)]) if steps else est


def factory(spec, cols):
    seeds = spec.get("seeds")
    if seeds:
        return lambda: models.SeedBag(lambda s: estimator(spec, cols, seed=s), seeds=tuple(seeds))
    return lambda: estimator(spec, cols)


def training_set(spec, X, y):
    """Training rows for a pipeline: the full-length battles, plus (optionally) the same battles
    truncated to fewer entries, with the full-length rows up-weighted."""
    aug = spec.get("aug")
    if not aug:
        return X, y, None
    A = load_truncated(tuple(aug["cuts"]))
    A = A.loc[A.index.isin(X.index), X.columns]
    ya = pd.Series(y, index=X.index).loc[A.index].to_numpy()
    w = np.r_[np.full(len(X), aug.get("w30", 1.0)), np.ones(len(A))]
    return pd.concat([X, A]), np.r_[y, ya], w
