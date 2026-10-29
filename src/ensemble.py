"""Blending of out-of-fold predictions: non-negative weights on the logits fitted by log-loss, and a
nested evaluation of the blend accuracy."""

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit, softmax
from sklearn.model_selection import StratifiedKFold

from .config import SEED


def _blend(P, w):
    return expit(np.tensordot(w, logit(np.clip(P, 1e-6, 1 - 1e-6)), axes=1))


def fit_weights(P, y):
    """P: (members, n) probabilities. Non-negative weights summing to one, minimising log-loss
    of the logit-averaged blend."""
    k = P.shape[0]
    L = logit(np.clip(P, 1e-6, 1 - 1e-6))

    def loss(z):
        p = np.clip(expit(softmax(z) @ L), 1e-9, 1 - 1e-9)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    z = minimize(loss, np.zeros(k), method="Nelder-Mead" if k < 3 else "L-BFGS-B").x
    return softmax(z)


def blend(P, w):
    return _blend(np.asarray(P), np.asarray(w))


def nested_accuracy(oofs, y, n_splits=5, seed=SEED):
    """oofs: (members, repeats, n). Weights fitted on 4/5 of the rows, scored on the rest."""
    oofs = np.asarray(oofs)
    accs = []
    for r in range(oofs.shape[1]):
        P = oofs[:, r]
        pred = np.zeros(len(y))
        for tr, va in StratifiedKFold(n_splits, shuffle=True, random_state=seed + r).split(P.T, y):
            w = fit_weights(P[:, tr], y[tr])
            pred[va] = blend(P[:, va], w)
        accs.append(np.mean((pred > 0.5) == y))
    return float(np.mean(accs))
