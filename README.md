# Pokémon Battle Prediction – FDS Kaggle Challenge 2026

Predicting the winner of Gen 2 OU Pokémon battles from Player 1's full team, Player 2's lead and the first
30 turns of the battle, with classical machine learning only.

With the three final submissions below, the best of which is F1, we ranked first on the private leaderboard.

## Final submissions

| | Submission | CV accuracy | CV log-loss | Public | Private |
|---|---|---|---|---|---|
| F1 | Blend of four models of pipeline A (`blend_v11`) | 72.08% (nested) | 0.5409 | 71.80% | 70.54% |
| F2 | Compact model of pipeline B (`compact_v5`, `ensemble_from_lr`) | 72.04% | 0.5395 | 71.00% | 70.09% |
| F3 | Logit blend of F1 and F2, with weight 0.55 on F2 | 72.32% (nested) | 0.5343 | 71.33% | 70.40% |

They were chosen on CV and diversity, since only the best selected score counts: F3 has the best CV of the three
and the two pipelines disagree on 12.2% of the out-of-fold predictions.

## Two pipelines

The group built two pipelines in parallel, exchanging ideas:

- **Pipeline A** (`src/`) is wide: hundreds of columns, a spline logistic regression and stacked gradient boosting
  (XGBoost, LightGBM, CatBoost), blended on the logits.
- **Pipeline B** (`compact/`) is compact: 18 selected columns and a logistic regression (LR) corrected by four
  gradient-boosting models that start from its log-odds.

In both, the Pokédex (types and base stats) comes from the battles and the test set is only predicted.

## Replaying the timelines

Both pipelines replay each timeline before building any feature, to rebuild what it omits:

- every entry is a decision point of Player 1 (P1): the state at the end of the turn, plus an extra entry when a P1
  Pokémon faints;
- the move field stores the *last* move used by the active Pokémon, so fresh moves are detected as changes of that
  field;
- Spikes, screens and weather are not recorded, so they are rebuilt from the moves;
- Player 2's (P2) lead is given separately and counts as revealed even when it switches out on turn 1 and never
  shows up in the timeline.

## Ten feature ideas

Both pipelines compare the two players through the same ideas. Pipeline A turns them into groups of columns, named
by prefix in `src/features/build.py`; pipeline B into single columns, listed in `compact/feature_catalog.py`.

| # | Idea | What it captures | Pipeline A | Pipeline B |
|---|---|---|---|---|
| 1 | Hidden knockouts | P2 faints missing from the records | `f_`, `src/faint.py` | `p2_fainted`, `p2_hidden_ko_unsure` |
| 2 | Material | HP, status and bulk (HP × defence) left to each side | `f_`, `x_`, `c_` | `smat_hp`, `material_healthy`, `phpw_spd` |
| 3 | Field | boosts, Spikes, Substitute | `a_`, `c_`, `w_` | `bst_sum`, `spikes_on_field`, `eff_substitute` |
| 4 | Actions | move classes, effectiveness, damage dealt, healing | `e_` | `b_stat_diff` |
| 5 | Team make-up | stats, items, types and key roles (Snorlax, sleepers, spinners, phazers) | `s_`, `w_`, bags | `snorlax_diff` |
| 6 | Duels | Gen 2 damage between the actives and a matchup matrix | `a_`, `m_` | none |
| 7 | Simulation | one-on-one duels played to the end, as the state alone ignores who beats whom | `d_`, `z_` | `sim_seen_left` |
| 8 | Momentum | earlier snapshots and a forecast of the next turns | `t_`, `q_` | `hpw_spd_t20` |
| 9 | Switching | who is under pressure, from voluntary-switch rates per pair of actives | `v_` | `p_switch_diff`, `switch_excess_all` |
| 10 | Team identity | reused teams as a proxy for the players, and partly revealed P2 teams completed from the known ones | `k_`, team effect | `tr_p1`, `tr_p2`, `te_pair_mean`, `hidden_bst` |

In pipeline A these ideas become about 400 columns, per player and per snapshot, plus 392 bag columns of species,
alive HP and revealed moves.

**Hidden knockouts.** P2's faints are recorded almost only when simultaneous with P1's: 0.5 per battle. Since P1's
faints are always recorded, a gradient-boosting detector is trained on P1's transitions, with the faint entry removed
so that they look like P2's, and applied to every P2 disappearance. It finds about one more faint per battle. The
detector of pipeline A reaches AUC 0.98; the one of pipeline B reaches 0.99 by also weighing the physical evidence
of each change of active Pokémon (speed order, expected damage on the leaver and on the incoming Pokémon, weather,
screens, end-of-turn damage). Without this repair the 18 columns of B fall from 71.98% to 69.48%.

**Simulation.** The remaining Pokémon fight a sequence of one-on-one duels: Gen 2 damage with the revealed moves,
alone or with the typical moves of each species learned from the training battles. The winner keeps its remaining HP
and the loser sends in its best answer. A second version of pipeline A also counts the turns still lost asleep or
frozen, always allows a STAB move, applies weather and screens to the duel of the actives, and is also run against
P2's unrevealed Pokémon guessed from the team co-occurrences of the training battles.

**Forecast.** In pipeline A, LightGBM regressors fitted on the training timelines map the state after 12 to 24
entries (the same features recomputed on the truncated timeline) to the change of the faint difference, the HP
difference and each player's faints over the next 5 or 10 entries, and are applied to the last decision point
(label-free, cross-fitted over the training battles).

**Switching.** Voluntary-switch rates per pair of actives, learned from the training battles, give each player's
chance to switch out in the current matchup and its switches beyond the expected ones.

**Team identity.** There are 4,167 distinct P1 teams in the 10,000 training battles, and 70% of the test battles use
a P1 team that also appears in training, so a rare team is a proxy for the player behind it.

- In pipeline A the team effect is the shrunk mean residual (outcome minus the cross-fitted LR probability) of the
  training battles played with a team: it counts positively when the team was P1's and negatively when it was P2's.
  P2's team is identified when all six Pokémon are revealed, or when the five revealed ones belong to a single known
  team. When only three to five are revealed, the battle is matched to all the known teams that contain them,
  weighted by how often each was played. A head-to-head term does the same for battles between the same two teams,
  and a raw rating, the shrunk win rate of each team from both sides, keeps the part of a player's strength already
  reflected in the battle state.
- In pipeline B the pool of known teams gives the win rates of P1's and P2's teams (`tr_p1`, `tr_p2`) and fills P2's
  unrevealed slots (`phpw_spd`), while `te_pair_mean` encodes the species pairs on the field and `hidden_bst` the
  expected strength of P2's hidden Pokémon.

All these encodings are cross-fitted, and the team-matching map is rebuilt from the training rows of each fold.

## Feature selection

**Pipeline A** keeps or rejects whole groups of columns on paired CV. Early drop-one-group ablations cost up to 0.7
points. The bag columns reach the boosters compressed into one cross-fitted logit.

**Pipeline B** selects single columns, in five steps of 14, 19, 19, 20 and 26 columns: a greedy forward-backward
search adds or removes one column at a time on paired log-loss, without lowering accuracy. A multi-seed revalidation
(6 seeds × 5 folds, corrected repeated k-fold t-test by Nadeau and Bengio) then drops the columns that do not help,
leaving 18. They cover all the ideas but the duels. Removing one group of columns from the LR of pipeline B costs:

| Columns removed | Accuracy | Log-loss |
|---|---|---|
| Material and knockouts | −1.76 points | +0.0185 |
| Team identity | −1.24 points | +0.0096 |
| Field | −1.19 points | +0.0082 |
| Simulation | −0.40 points | +0.0035 |
| Switching | −0.35 points | +0.0021 |

## Models

**Pipeline A.** A logistic regression with splines on the key state features, plus LightGBM, XGBoost and CatBoost
tuned with Optuna. The tree models receive the bag logit, a cross-fitted logit of a logistic regression on all
features, and the team-effect features. F1 averages the logits of the four models with non-negative weights fitted
on out-of-fold predictions (CatBoost 0.37, LightGBM 0.27, spline LR 0.21, XGBoost 0.15).

**Pipeline B.** An L1 logistic regression on the 18 columns already reaches 71.98%, while boosters trained from
scratch with default parameters reach 70.91% to 71.58%. So four boosters (scikit-learn gradient boosting, CatBoost,
LightGBM, XGBoost) start from its log-odds and only learn a small correction. F2 averages their probabilities, which
lowers the log-loss by 0.003.

**F3** blends F1 and F2 on their logits. `compact/features/v11_bridge.py` trains `blend_v11` of `src/` on the same
battles, and the weight of F2 (0.55) is chosen by log-loss on the CV.

## Validation

- Decisions used repeated stratified 5-fold CV with duplicates grouped, comparing changes fold by fold. Accuracy is
  the target metric, with log-loss and AUC as tie-breakers.
- All the numbers here share the same three repeats (`src/cv.py`, seeds 42 to 44). Those of pipeline B were measured
  on these folds with the final code. `python main.py evaluate --n-seeds 3` in `compact/` runs the same protocol, but
  it groups duplicates on the whole record, so its folds differ slightly.
- Label-dependent steps are cross-fitted inside the training folds, blend weights are evaluated with a nested split
  and Optuna used a different split than the one used to compare models.
- With shuffled labels accuracy is 50.4% for LightGBM in pipeline A, and 50.6% (LR) and 49.8% (F2) in pipeline B. On
  fresh folds (5×2, fixed weights) F1, F2 and F3 reach 72.04%, 72.01% and 72.43%.
- Selection still biases CV: repeating the column selection of pipeline B inside the folds costs 0.35 points (one
  repeat). Fresh folds cannot show this, because they reuse the same battles.

**Against the leaderboards.** On 14 submissions the public score stays within one standard error of CV (1.2 points
on 1,500 battles), 0.45 points lower on average, but it is uncorrelated with CV over the last ten: it was only a
sanity check. On the private split (3,500 battles, standard error 0.8) the three finals fall 1.5 to 2.0 points below
CV. Post-hoc KS tests on 435 columns find no train–test shift, so noise and selection on the same battles likely
explain it. F1 leads F3 and F2 by 7 and 12 public and 5 and 16 private battles, within one paired standard error.

## CV results

Pipeline A, each row adding to the previous one from the tuned CatBoost onwards (`experiments/log.csv`):

| Step | Pipeline | Accuracy | Log-loss |
|---|---|---|---|
| Logistic regression, first feature set | `lr_base` | 69.32% | 0.5736 |
| LightGBM with the stacked bag logit | `lgbm_stack` | 70.31% | 0.5635 |
| Spline logistic regression | `lrs_mk_v4ref` | 70.55% | 0.5571 |
| Tuned CatBoost with the stacked logits | `cat_t2` | 70.72% | 0.5574 |
| + team effect, head-to-head term | `cat_h2h` | 71.14% | 0.5521 |
| + simulation, switching behaviour | `cat_sim` | 71.55% | 0.5497 |
| + simulation v2, field and team condition | `cat_sim3` | 71.65% | 0.5474 |
| + soft matching of partial P2 teams, forecast, P2 lead | `cat_v10` | 71.88% | 0.5446 |
| + raw team ratings | `cat_v11` | 72.04% | 0.5421 |
| F1: blend of spline LR, XGBoost, LightGBM, CatBoost | `blend_v11` | 72.08% (nested) | 0.5409 |

Pipeline B, with the columns left by the revalidation:

| Step | Feature set | Accuracy | Log-loss |
|---|---|---|---|
| HP difference alone, LR | `hp30` | 65.25% | 0.6167 |
| LR, 11 columns | `compact` | 68.10% | 0.5878 |
| + team encodings, simulation | `compact_plus` | 69.02% | 0.5781 |
| + knockout repair, re-selection | `compact_v4` | 71.50% | 0.5486 |
| + team pool, re-selection | `compact_v5` | 71.98% | 0.5425 |
| F2: boosters from the LR logit | `compact_v5` | 72.04% | 0.5395 |

## What did not help

- time-share and count variants of the bags;
- the expected composition of P2's unrevealed team;
- pairwise interactions in the logistic regression;
- separate stacked logits for each bag;
- a spline-based stacked logit;
- training-set augmentation with truncated battles;
- monotone constraints and seed bagging;
- re-tuning after the team effect;
- separate P1 / P2 team terms;
- critical hits and usage-based typical moves in the simulation;
- a joint Bradley–Terry estimate of the team strengths;
- a pairwise coverage score on top of the matchup matrix (best answers and worst matchups).

`experiments/log.csv` records every cross-validation run of pipeline A with its full configuration, and
`configs/pipelines.json` keeps the main pipelines and one representative of each experiment family.

## Regenerating the final submissions

The Kaggle notebook clones this repository and regenerates the three selected submissions:

```bash
pip install -r requirements.txt
python -m src.cli features                      # build and cache the features (about 6 minutes)
python -m src.cli submit --blend blend_v11      # F1

cd compact
python main.py train --features compact_v5 --models ensemble_from_lr --out runs     # F2
python main.py predict --run runs/compact_v5__ensemble_from_lr --out ../submissions/compact.csv
python main.py train --features compact_v5_v11 --models blend_v11 --params blend_params.json --out runs   # F3
python main.py predict --run runs/compact_v5_v11__blend_v11-tuned --out ../submissions/compact_blend.csv
```

- The data folder is found automatically: the local folder `fds-pokemon-battles-prediction-2026/` or the Kaggle
  input directory. It can be set with `FDS_DATA_DIR`.
- `FDS_N_JOBS` sets the number of threads (default 4).
