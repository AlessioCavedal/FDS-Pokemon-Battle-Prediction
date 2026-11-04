"""Model factory and custom pipeline steps: cross-fitted stacked logits, the team effect, the spline logistic
regression and seed bagging."""

from collections import defaultdict

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin, OneToOneFeatureMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler
from xgboost import XGBClassifier

from .config import N_JOBS, SEED

SPLINE_KEYS = ["f_d_fainted", "f_d_hp_sum", "f_p1_fainted", "f_p2_fainted", "f_p1_hp_sum", "f_p2_hp_sum",
               "f_d_strength", "t_hd_mean", "t_fd_mean", "a_ttk_adv", "m_win_mean", "f_d_alive", "a_a1_hp", "a_a2_hp"]
BAG_PREFIXES = ("bs_", "bh_", "bm_", "bt_", "bc_")

DEFAULTS = {
    "lr": {"C": 0.05},
    "lr_spline": {"C": 0.0007, "n_knots": 4, "bag_scale": 1.0},
    "lgbm": {"n_estimators": 600, "learning_rate": 0.02, "num_leaves": 15, "min_child_samples": 40,
             "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.5, "reg_lambda": 5.0},
    "xgb": {"n_estimators": 600, "learning_rate": 0.02, "max_depth": 4, "min_child_weight": 5,
            "subsample": 0.8, "colsample_bytree": 0.5, "reg_lambda": 5.0},
    "cat": {"iterations": 1500, "learning_rate": 0.03, "depth": 6, "l2_leaf_reg": 5.0},
    "et": {"n_estimators": 800, "max_features": 0.3, "min_samples_leaf": 5},
    "rf": {"n_estimators": 800, "max_features": 0.2, "min_samples_leaf": 5},
}


def make(name, params=None, seed=SEED):
    p = {**DEFAULTS.get(name, {}), **(params or {})}
    if name == "lr":
        return Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                         ("lr", LogisticRegression(max_iter=5000, **p))])
    if name == "lr_spline":
        return spline_lr(**p)
    if name == "lgbm":
        return LGBMClassifier(random_state=seed, n_jobs=N_JOBS, deterministic=True, force_row_wise=True,
                              verbose=-1, **p)
    if name == "xgb":
        return XGBClassifier(random_state=seed, n_jobs=N_JOBS, tree_method="hist", eval_metric="logloss", **p)
    if name == "cat":
        return CatBoostClassifier(random_seed=seed, thread_count=N_JOBS, verbose=0, allow_writing_files=False, **p)
    if name == "et":
        return Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("et", ExtraTreesClassifier(random_state=seed, n_jobs=N_JOBS, **p))])
    if name == "rf":
        return Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("rf", RandomForestClassifier(random_state=seed, n_jobs=N_JOBS, **p))])
    raise ValueError(name)


MONOTONE = {"stack_logit_full": 1, "stack_logit_0": 1, "f_d_fainted": -1, "f_p1_fainted": -1, "f_p2_fainted": 1,
            "f_d_hp_sum": 1, "f_p1_hp_sum": 1, "f_p2_hp_sum": -1, "f_d_alive": 1, "f_d_fainted_hard": 1,
            "f_d_hp_hard": 1, "f_d_strength": 1}


class Monotone(BaseEstimator, ClassifierMixin):
    """Sets monotone constraints of a LightGBM/XGBoost model from column names at fit time."""

    def __init__(self, model, signs=None):
        self.model, self.signs = model, signs

    def fit(self, X, y, sample_weight=None):
        signs = self.signs or MONOTONE
        cons = [signs.get(c, 0) for c in X.columns]
        if isinstance(self.model, XGBClassifier):
            cons = "(" + ",".join(map(str, cons)) + ")"
        self.model_ = self.model.set_params(monotone_constraints=cons).fit(X, y, sample_weight=sample_weight)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)


class PrefixScale(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Multiplies standardized columns with the given prefixes, i.e. a stronger L2 penalty on them."""

    def __init__(self, prefixes=BAG_PREFIXES, scale=1.0):
        self.prefixes, self.scale = prefixes, scale

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        if self.scale == 1.0:
            return X
        X = X.copy()
        cols = [c for c in X.columns if c.startswith(self.prefixes)]
        X[cols] = X[cols] * self.scale
        return X


def spline_lr(C=0.0007, n_knots=4, bag_scale=1.0, keys=None):
    keys = keys or SPLINE_KEYS
    spl = Pipeline([("imp", SimpleImputer(strategy="median")), ("s", SplineTransformer(n_knots=n_knots, degree=3))])
    pipe = Pipeline([
        ("ct", ColumnTransformer([("spl", spl, keys)], remainder="passthrough", verbose_feature_names_out=False)),
        ("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
        ("scale", PrefixScale(scale=bag_scale)),
        ("lr", LogisticRegression(C=C, max_iter=5000)),
    ])
    return pipe.set_output(transform="pandas")


class StackedLogit(BaseEstimator, TransformerMixin):
    """Replaces a block of columns with the logit of an L2 logistic regression fitted on them.
    fit_transform returns cross-fitted values, transform uses the model fitted on all rows."""

    def __init__(self, columns, C=0.05, n_splits=5, seed=SEED, name="stack_logit", keep=False, spline_keys=None):
        self.columns, self.C, self.n_splits, self.seed, self.name, self.keep = columns, C, n_splits, seed, name, keep
        self.spline_keys = spline_keys

    def _model(self):
        if self.spline_keys:
            return spline_lr(C=self.C, keys=[k for k in self.spline_keys if k in self.columns])
        return Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                         ("lr", LogisticRegression(C=self.C, max_iter=5000))])

    def fit(self, X, y):
        self.model_ = self._model().fit(X[self.columns], y)
        return self

    def fit_transform(self, X, y=None, **kw):
        Z = X[self.columns]
        out = np.zeros(len(X))
        if X.index.is_unique:
            splits = StratifiedKFold(self.n_splits, shuffle=True, random_state=self.seed).split(Z, y)
        else:
            splits = StratifiedGroupKFold(self.n_splits, shuffle=True, random_state=self.seed).split(Z, y, X.index)
        for tr, va in splits:
            m = self._model().fit(Z.iloc[tr], y[tr])
            out[va] = m.decision_function(Z.iloc[va])
        self.model_ = self._model().fit(Z, y)
        return self._attach(X, out)

    def transform(self, X):
        return self._attach(X, self.model_.decision_function(X[self.columns]))

    def _attach(self, X, score):
        X = X if self.keep else X.drop(columns=self.columns)
        return X.assign(**{self.name: score})


class TeamEffect(BaseEstimator, TransformerMixin):
    """Shrunk mean residual (w.r.t. the cross-fitted logit `base`) of the training battles played with the
    same team, a proxy for the player behind it. A battle counts for its P1 team and, with opposite sign,
    for its P2 team when known; with `link`, a P2 team with five revealed species is matched to the only
    training team containing them. With `soft`, a P2 team with `soft_min` to five revealed species is matched
    to all the training teams that contain them, weighted by how often each was played: such battles spread
    their residual over these candidates and get their weighted effect (`tm_soft`, also added to `tm` when
    the P2 team is not known otherwise). With `joint`, the strengths are instead fitted together
    (Bradley-Terry style, see _joint). fit_transform is cross-fitted, transform uses all training rows."""

    def __init__(self, base="stack_logit_full", a=5.0, p2=True, link=False, pair=False, split=False, joint=False,
                 lam=1.0, lam_pair=None, n_splits=5, seed=SEED, drop_base=False, soft=False, soft_min=3):
        self.base, self.a, self.p2, self.link, self.pair, self.split = base, a, p2, link, pair, split
        self.joint, self.lam, self.lam_pair = joint, lam, lam_pair
        self.n_splits, self.seed, self.drop_base = n_splits, seed, drop_base
        self.soft, self.soft_min = soft, soft_min

    @staticmethod
    def _pair_key(a, b):
        return ((a, b), 1.0) if a <= b else ((b, a), -1.0)

    def _joint(self, k1, k2, base, y):
        """One strength per team (and per pairing): L2-penalised logistic regression with the base logit as
        offset, P1 team +1, P2 team -1, so each team is credited relative to the opponents it met."""
        index, rows, cols, vals = {}, [], [], []
        for i, (a, b) in enumerate(zip(k1, k2)):
            terms = [(a, 1.0)]
            if not np.isnan(b):
                if self.p2:
                    terms.append((b, -1.0))
                if self.pair:
                    terms.append(self._pair_key(a, b))
            for key, v in terms:
                rows.append(i)
                cols.append(index.setdefault(key, len(index)))
                vals.append(v)
        A = csr_matrix((vals, (rows, cols)), shape=(len(k1), len(index)))
        lam_pair = self.lam if self.lam_pair is None else self.lam_pair
        pen = np.array([lam_pair if isinstance(k, tuple) else self.lam for k in index])

        def loss(t):
            z = base + A @ t
            return np.sum(np.logaddexp(0, z) - y * z) + 0.5 * pen @ (t * t), A.T @ (expit(z) - y) + pen * t

        return dict(zip(index, minimize(loss, np.zeros(len(index)), jac=True, method="L-BFGS-B").x))

    def _links(self, X):
        links = defaultdict(set)
        if self.link:
            for side in ("1", "2"):
                subs = X[[f"k_p{side}sub{k}" for k in range(6)]].to_numpy()
                for team, row in zip(X[f"k_p{side}team"].to_numpy(), subs):
                    if not np.isnan(team):
                        for sub in row:
                            links[sub].add(team)
        return links

    def _keys(self, X, links):
        k1, k2 = X["k_p1team"].to_numpy(), X["k_p2team"].to_numpy().copy()
        if self.link:
            part = X["k_p2part"].to_numpy()
            for i in np.flatnonzero(~np.isnan(part)):
                cands = links.get(part[i], set()) - {k1[i]}
                if len(cands) == 1:
                    k2[i] = cands.pop()
        return k1, k2

    def _fit(self, X, y):
        links = self._links(X)
        k1, k2 = self._keys(X, links)
        base = X[self.base].to_numpy()
        theta = self._joint(k1, k2, base, y) if self.joint else None
        r = y - expit(base)
        s, c = defaultdict(float), defaultdict(int)
        for a, b, ri in zip(k1, k2, r):
            s[a] += ri
            c[a] += 1
            if not np.isnan(b):
                if self.p2:
                    s[b] -= ri
                    c[b] += 1
                if self.pair:
                    s[a, b] += ri
                    c[a, b] += 1
                    s[b, a] -= ri
                    c[b, a] += 1
        return s, c, links, theta, self._soft_fit(X, r) if self.soft else None

    @staticmethod
    def _species(X):
        p1 = X[[f"k_p1sp{k}" for k in range(6)]].to_numpy()
        p2 = X[[f"k_p2sp{k}" for k in range(6)]].to_numpy()
        return [frozenset(row) for row in p1], [frozenset(v for v in row if not np.isnan(v)) for row in p2]

    @staticmethod
    def _candidates(index, revealed, own):
        cands = None
        for sp in sorted(revealed):
            cands = set(index.get(sp, ())) if cands is None else cands & index.get(sp, set())
            if not cands:
                return []
        return sorted(cands - {own}, key=sorted)

    def _partial(self, revealed):
        return self.soft_min <= len(revealed) < 6

    def _soft_fit(self, X, r):
        """Team strengths over sets of species: full teams as usual, then every partially revealed P2 team
        shares its residual among the candidates in proportion to their counts."""
        sp1, sp2 = self._species(X)
        s, c = defaultdict(float), defaultdict(float)
        for t1, t2, ri in zip(sp1, sp2, r):
            s[t1] += ri
            c[t1] += 1
            if len(t2) == 6:
                s[t2] -= ri
                c[t2] += 1
        index = defaultdict(set)
        for team in c:
            for sp in team:
                index[sp].add(team)
        add_s, add_c = defaultdict(float), defaultdict(float)
        for t1, t2, ri in zip(sp1, sp2, r):
            cands = self._candidates(index, t2, t1) if self._partial(t2) else []
            if cands:
                w = np.array([c[t] for t in cands])
                for team, wt in zip(cands, w / w.sum()):
                    add_s[team] -= wt * ri
                    add_c[team] += wt
        for team in add_s:
            s[team] += add_s[team]
            c[team] += add_c[team]
        return s, c, index

    def _soft_encode(self, X, state):
        s, c, index = state
        sp1, sp2 = self._species(X)
        effect, count = np.zeros(len(X)), np.zeros(len(X))
        for i, (t1, t2) in enumerate(zip(sp1, sp2)):
            cands = self._candidates(index, t2, t1) if self._partial(t2) else []
            if cands:
                w = np.array([c[t] for t in cands])
                e = np.array([s[t] / (c[t] + self.a) for t in cands])
                effect[i], count[i] = -(w @ e) / w.sum(), w.sum()
        return effect, count

    def _encode(self, X, state):
        s, c, links, theta, soft = state
        k1, k2 = self._keys(X, links)
        known = ~np.isnan(k2)
        n1 = np.array([c.get(k, 0) for k in k1], dtype=float)
        n2 = np.array([c.get(k, 0) if ok else 0 for k, ok in zip(k2, known)], dtype=float)
        if self.joint:
            e1 = np.array([theta.get(k, 0.0) for k in k1])
            e2 = np.array([theta.get(k, 0.0) if ok else 0.0 for k, ok in zip(k2, known)])
        else:
            e1 = np.array([s.get(k, 0.0) for k in k1]) / (n1 + self.a)
            e2 = np.array([s.get(k, 0.0) if ok else 0.0 for k, ok in zip(k2, known)]) / (n2 + self.a)
        out = {"tm": e1 - (e2 if self.p2 else 0.0), "tm_n1": n1, "tm_n2": n2}
        if self.split:
            out.update(tm_e1=e1, tm_e2=e2)
        if self.pair:
            pairs = list(zip(k1, k2))
            n = np.array([c.get(p, 0) if ok else 0 for p, ok in zip(pairs, known)], dtype=float)
            if self.joint:
                keys = [self._pair_key(a, b) for a, b in pairs]
                out["tm_pair"] = np.array([theta.get(k, 0.0) * sg if ok else 0.0 for (k, sg), ok in zip(keys, known)])
            else:
                out["tm_pair"] = np.array([s.get(p, 0.0) if ok else 0.0 for p, ok in zip(pairs, known)]) / (n + self.a)
            out["tm_npair"] = n
        if self.soft:
            out["tm_soft"], out["tm_nsoft"] = self._soft_encode(X, soft)
            out["tm"] = out["tm"] + np.where(n2 == 0, out["tm_soft"], 0.0)
        return out

    def fit(self, X, y):
        self.state_ = self._fit(X, np.asarray(y))
        return self

    def fit_transform(self, X, y=None, **kw):
        y = np.asarray(y)
        cols = {}
        if X.index.is_unique:
            splits = StratifiedKFold(self.n_splits, shuffle=True, random_state=self.seed).split(X, y)
        else:
            splits = StratifiedGroupKFold(self.n_splits, shuffle=True, random_state=self.seed).split(X, y, X.index)
        for tr, va in splits:
            for name, v in self._encode(X.iloc[va], self._fit(X.iloc[tr], y[tr])).items():
                cols.setdefault(name, np.zeros(len(X)))[va] = v
        self.state_ = self._fit(X, y)
        return self._attach(X, cols)

    def transform(self, X):
        return self._attach(X, self._encode(X, self.state_))

    def _attach(self, X, cols):
        drop = [c for c in X.columns if c.startswith("k_")] + ([self.base] if self.drop_base else [])
        return pd.concat([X.drop(columns=drop), pd.DataFrame(cols, index=X.index)], axis=1)


def stack_steps(blocks, C=0.05, seed=SEED, full_C=None, full_cols=None, full_spline_keys=None):
    steps = []
    if full_C:
        steps.append(("stack_full", StackedLogit(full_cols, C=full_C, seed=seed, name="stack_logit_full", keep=True,
                                                 spline_keys=full_spline_keys)))
    steps += [(f"stack{k}", StackedLogit(cols, C=C, seed=seed, name=f"stack_logit_{k}"))
              for k, cols in enumerate(blocks)]
    return steps


def fit_weighted(est, X, y, w=None):
    if w is None:
        return est.fit(X, y)
    if isinstance(est, Pipeline):
        return est.fit(X, y, **{f"{est.steps[-1][0]}__sample_weight": w})
    return est.fit(X, y, sample_weight=w)


class SeedBag(BaseEstimator, ClassifierMixin):
    def __init__(self, factory, seeds=(0, 1, 2, 3, 4)):
        self.factory, self.seeds = factory, seeds

    def fit(self, X, y, sample_weight=None):
        self.models_ = [fit_weighted(self.factory(seed), X, y, sample_weight) for seed in self.seeds]
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.models_], axis=0)
