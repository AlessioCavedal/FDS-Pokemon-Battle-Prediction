"""Command line entry point: `python -m src.cli features | cv | blend | submit`."""

import argparse
import json

import numpy as np
import pandas as pd

from . import cv, ensemble, pipelines
from .config import CONFIG_DIR, SUB_DIR
from .data import load_sample_submission
from .features.build import load
from .models import fit_weighted


def _data():
    d = load()
    return d["X_train"], d["X_test"], d["y"].to_numpy(), d["groups"].to_numpy()


def cmd_features(a):
    d = load(rebuild=True)
    print(d["info"], d["X_train"].shape, d["X_test"].shape)


def cmd_cv(a):
    X, _, y, g = _data()
    folds = cv.make_folds(y, g, n_splits=a.splits, n_repeats=a.repeats)
    all_specs = pipelines.specs()
    for name in a.pipelines:
        spec = all_specs[name]
        cols = pipelines.columns(spec, X.columns)
        ts = (lambda Xt, yt, spec=spec: pipelines.training_set(spec, Xt, yt)) if spec.get("aug") else None
        res, _ = cv.run(pipelines.factory(spec, cols), X[cols], y, folds, name=name, notes=a.notes, params=spec,
                        train_set=ts)
        print(f"{name:24s} acc {res['acc']:.4f}  rep_sd {res['acc_rep_std']:.4f}  ll {res['logloss']:.4f}  "
              f"auc {res['auc']:.4f}  nf {res['n_features']}  {res['seconds']}s", flush=True)


def cmd_blend(a):
    _, _, y, _ = _data()
    oofs = np.stack([np.load(cv.OOF_DIR / f"{m}.npy") for m in a.members])
    w = ensemble.fit_weights(oofs.reshape(len(a.members), -1), np.tile(y, oofs.shape[1]))
    full = cv.evaluate(np.stack([ensemble.blend(oofs[:, r], w) for r in range(oofs.shape[1])]), y)
    nested = ensemble.nested_accuracy(oofs, y)
    print("weights", dict(zip(a.members, np.round(w, 4))))
    print(f"blend acc {full['acc']:.4f}  ll {full['logloss']:.4f}  auc {full['auc']:.4f}  nested acc {nested:.4f}")
    if a.save:
        path = CONFIG_DIR / "blends.json"
        blends = json.loads(path.read_text()) if path.exists() else {}
        blends[a.save] = {"members": dict(zip(a.members, map(float, w))), "cv_acc": full["acc"],
                          "nested_acc": nested}
        path.write_text(json.dumps(blends, indent=2))


def predict_test(members):
    X, Xt, y, _ = _data()
    all_specs = pipelines.specs()
    probs = []
    for name in members:
        spec = all_specs[name]
        cols = pipelines.columns(spec, X.columns)
        Xf, yf, wf = pipelines.training_set(spec, X[cols], y)
        model = fit_weighted(pipelines.factory(spec, cols)(), Xf, yf, wf)
        probs.append(model.predict_proba(Xt[cols])[:, 1])
        print(f"fitted {name}", flush=True)
    return Xt.index, np.stack(probs)


def write_submission(index, prob, name):
    sub = load_sample_submission()
    pred = pd.Series((prob > 0.5).astype(int), index=index)
    sub["player_won"] = sub["battle_id"].map(pred).astype(int)
    path = SUB_DIR / f"{name}.csv"
    sub.to_csv(path, index=False)
    print(f"wrote {path}  rows {len(sub)}  mean {sub['player_won'].mean():.4f}")
    return path


def cmd_submit(a):
    if a.blend:
        spec = json.loads((CONFIG_DIR / "blends.json").read_text())[a.blend]
        members, weights = list(spec["members"]), np.array(list(spec["members"].values()))
        index, P = predict_test(members)
        prob = ensemble.blend(P, weights)
        name = a.name or a.blend
    else:
        index, P = predict_test([a.pipeline])
        prob = P[0]
        name = a.name or a.pipeline
    write_submission(index, prob, name)
    np.save(SUB_DIR / f"{name}_proba.npy", prob)


def main():
    p = argparse.ArgumentParser(prog="python -m src.cli")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("features").set_defaults(fn=cmd_features)
    c = sub.add_parser("cv")
    c.add_argument("pipelines", nargs="+")
    c.add_argument("--splits", type=int, default=5)
    c.add_argument("--repeats", type=int, default=3)
    c.add_argument("--notes", default="")
    c.set_defaults(fn=cmd_cv)
    b = sub.add_parser("blend")
    b.add_argument("members", nargs="+")
    b.add_argument("--save")
    b.set_defaults(fn=cmd_blend)
    s = sub.add_parser("submit")
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument("--pipeline")
    g.add_argument("--blend")
    s.add_argument("--name")
    s.set_defaults(fn=cmd_submit)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
