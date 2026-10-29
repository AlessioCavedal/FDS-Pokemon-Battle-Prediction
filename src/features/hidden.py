"""Expected composition of the unrevealed part of Player 2's team from team co-occurrences in the
training battles (an experiment not used by the final pipelines)."""

import numpy as np


class TeamModel:
    """Naive-Bayes co-occurrence model of complete teams, fitted on player 1 teams of training battles;
    gives the expected composition of player 2's unrevealed slots."""

    def __init__(self, battles, alpha=1.0, temperature=0.5):
        names = sorted({p["name"] for b in battles for p in b.team})
        self.index = {n: i for i, n in enumerate(names)}
        T = np.zeros((len(battles), len(names)))
        for k, b in enumerate(battles):
            for p in b.team:
                T[k, self.index[p["name"]]] = 1
        self.names = names
        cnt = T.sum(0)
        self.log_prior = np.log((cnt + alpha) / (len(battles) + 2 * alpha))
        co = T.T @ T
        cond = (co + alpha) / (cnt[:, None] + 2 * alpha)
        marg = (cnt + alpha) / (len(battles) + 2 * alpha)
        self.log_lift = np.log(cond / marg[None, :])
        self.temperature = temperature

    def expected_unrevealed(self, revealed, n_hidden):
        if n_hidden <= 0:
            return {}
        idx = [self.index[r] for r in revealed if r in self.index]
        score = self.log_prior + self.temperature * self.log_lift[:, idx].sum(1)
        score[idx] = -np.inf
        q = np.exp(score - score[np.isfinite(score)].max())
        q /= q.sum()
        e = 1 - (1 - q) ** n_hidden
        e *= n_hidden / e.sum()
        e = np.minimum(e, 1.0)
        return {self.names[i]: float(e[i]) for i in np.flatnonzero(e > 1e-3)}


def hidden_features(b, model, vocab, t1, t2):
    rev2 = list(b.sides["p2"].revealed)
    n_hidden = 6 - len(rev2)
    exp = model.expected_unrevealed(rev2, n_hidden)
    f = {}
    team = {p["name"] for p in b.team}
    for s in vocab.species:
        e = exp.get(s, 0.0)
        p1 = float(s in team)
        f[f"bf_{s}"] = p1 - (float(s in t2["hp"]) + e)
        h1 = (1 - t1["pf"][s]) * t1["hp"][s] if s in team else 0.0
        h2 = (1 - t2["pf"][s]) * t2["hp"][s] if s in t2["hp"] else 0.0
        f[f"bg_{s}"] = h1 - (h2 + e)
    bst = {s: b.dex[s]["bst"] for s in exp if s in b.dex}
    f["u_exp_bst"] = sum(e * bst.get(s, 0) for s, e in exp.items())
    f["u_exp_max_prob"] = max(exp.values(), default=0.0)
    return f
