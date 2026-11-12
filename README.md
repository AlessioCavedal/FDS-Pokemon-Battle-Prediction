# Pokémon Battle Prediction – FDS Kaggle Challenge 2026

Winning solution of the Sapienza University FDS Kaggle Challenge 2026: **1st of 52 teams** (about 200 students) on
the private leaderboard.

We predict the winner of Gen 2 OU Pokémon battles from Player 1's full team, Player 2's lead and the first 30 turns
of the battle, with classical machine learning only. The one-page [paper](Report.pdf) summarises the method, the
experiments and the results.

## Final submissions

| | Submission | CV accuracy | CV log-loss | Public | Private |
|---|---|---|---|---|---|
| F1 | Blend of four models of pipeline A (`blend_v11`) | 72.08% (nested) | 0.5409 | 71.80% | 70.54% |
| F2 | Compact model of pipeline B (`compact_v5`, `ensemble_from_lr`) | 72.04% | 0.5395 | 71.00% | 70.09% |
| F3 | Logit blend of F1 and F2, with weight 0.55 on F2 | 72.32% (nested) | 0.5343 | 71.33% | 70.40% |

F1 gave the winning score. The three were chosen on CV and diversity, since only the best selected score counts: F3
has the best CV of the three and the two pipelines disagree on 12.2% of the out-of-fold predictions.

## Two pipelines

The group built two pipelines in parallel, exchanging ideas:

- **Pipeline A** (`src/`) is wide: hundreds of columns, a spline logistic regression and stacked gradient boosting
  (XGBoost, LightGBM, CatBoost), blended on the logits.
- **Pipeline B** (`compact/`) is compact: 18 selected columns and a logistic regression (LR) corrected by four
  gradient-boosting models that start from its log-odds.

In both, the Pokédex (types and base stats) comes from the battles and the test set is only predicted. Each replays
the timelines before building any feature, to detect fresh moves and to rebuild what the records omit, such as
Spikes, screens, weather and most of Player 2's (P2) faints.

## Ten feature ideas

Both pipelines compare Player 1 (P1) and P2 through the same ideas. Pipeline A turns them into groups of columns,
named by prefix in `src/features/build.py`; pipeline B into single columns, listed in `compact/feature_catalog.py`.

| # | Idea | What it captures |
|---|---|---|
| 1 | Hidden knockouts | P2 faints missing from the records |
| 2 | Material | HP, status and bulk (HP × defence) left to each side |
| 3 | Field | boosts, Spikes, Substitute |
| 4 | Actions | move classes, effectiveness, damage dealt, healing |
| 5 | Team make-up | stats, items, types and key roles (Snorlax, sleepers, spinners, phazers) |
| 6 | Duels | Gen 2 damage between the actives and a matchup matrix |
| 7 | Simulation | one-on-one duels played to the end, as the state alone ignores who beats whom |
| 8 | Momentum | earlier snapshots and a forecast of the next turns |
| 9 | Switching | who is under pressure, from voluntary-switch rates per pair of actives |
| 10 | Team identity | reused teams as a proxy for the players, and partly revealed P2 teams completed from the known ones |

**Hidden knockouts.** P2's faints are recorded almost only when simultaneous with P1's (0.5 per battle). P1's faints
are always recorded, so a gradient-boosting detector trained on P1's transitions finds about one more per battle (AUC
0.98, or 0.99 in pipeline B, which also weighs the physical evidence of each switch). Without this repair the 18
columns of B fall from 71.98% to 69.48%.

**Simulation and forecast.** The remaining Pokémon fight a sequence of one-on-one duels with the Gen 2 damage
formula: the winner keeps its HP and the loser sends in its best answer. Pipeline A also forecasts the next turns,
with regressors trained on truncated timelines.

**Team identity.** There are 4,167 distinct P1 teams in the 10,000 training battles, and 70% of the test battles use
a P1 team seen in training, so a rare team is a proxy for its player. Pipeline A encodes the shrunk residual of each
team against a cross-fitted LR, a head-to-head term and a raw rating; pipeline B the win rates of both teams.
Partly revealed P2 teams are matched to the known teams that contain them.

In pipeline A these ideas become about 400 columns, per player and per snapshot, plus 392 bag columns of species,
alive HP and revealed moves. Each module's docstring explains the details.

## Selection and models

**Pipeline A** keeps or rejects whole groups of columns on paired CV. F1 blends the logits of a spline LR and of
XGBoost, LightGBM and CatBoost tuned with Optuna; the trees see the bags as one cross-fitted logit.

**Pipeline B** selects single columns: a greedy forward-backward search on paired log-loss leaves 26, and a
multi-seed revalidation with a corrected t-test leaves 18, covering all the ideas but the duels. LR reaches 71.98% on
them and boosters from scratch 70.91% to 71.58%, so the four boosters of F2 start from its log-odds and only learn a
correction.

**F3** blends F1 and F2 on their logits, with the weight of F2 chosen by log-loss.

## Validation

Decisions used repeated stratified 5-fold CV with duplicates grouped, comparing changes fold by fold. All the numbers
here use the same three repeats (`src/cv.py`); `python main.py evaluate --n-seeds 3` in `compact/` groups duplicates
slightly differently, so its folds are not identical. Label-dependent steps are cross-fitted inside the training
folds, blend weights are nested and Optuna used a separate split. Shuffled labels score about 50%, and on fresh folds
F1, F2 and F3 reach 72.0%, 72.0% and 72.4%. Selection still biases CV: repeating the selection of pipeline B inside
the folds costs 0.35 points (one repeat).

On 14 submissions the public score stays within one standard error of CV (1.2 points on 1,500 battles), 0.45 lower on
average, but is uncorrelated over the last ten: it was only a sanity check. On the private split the three finals
fall 1.5 to 2.0 points below CV (standard error 0.8). Post-hoc KS tests on 435 columns find no train–test shift, so
noise and selection on the same battles likely explain it. F1 leads F3 and F2 by 5 and 16 private battles, within one
paired standard error.

## CV results

Rows with + add to the row above.

| Model | Name | Accuracy | Log-loss |
|---|---|---|---|
| A: LR, first feature set | `lr_base` | 69.32% | 0.5736 |
| A: LightGBM with the bag logit | `lgbm_stack` | 70.31% | 0.5635 |
| A: tuned CatBoost with the stacked logits | `cat_t2` | 70.72% | 0.5574 |
| + team effect, head-to-head term | `cat_h2h` | 71.14% | 0.5521 |
| + simulation, switching, field and team condition | `cat_sim3` | 71.65% | 0.5474 |
| + soft matching of partial P2 teams, forecast, P2 lead | `cat_v10` | 71.88% | 0.5446 |
| + raw team ratings | `cat_v11` | 72.04% | 0.5421 |
| **F1**: blend of four models | `blend_v11` | 72.08% (nested) | 0.5409 |
| B: LR, 11 columns | `compact` | 68.10% | 0.5878 |
| + team encodings, simulation | `compact_plus` | 69.02% | 0.5781 |
| + knockout repair, re-selection | `compact_v4` | 71.50% | 0.5486 |
| + team pool, re-selection | `compact_v5` | 71.98% | 0.5425 |
| **F2**: boosters from the LR logit | `ensemble_from_lr` | 72.04% | 0.5395 |
| **F3**: logit blend of F1 and F2 | `blend_v11` in `compact/` | 72.32% (nested) | 0.5343 |

Seed bagging, monotone constraints, training on truncated battles, pairwise LR interactions, separate stacks of the
bags, Bradley–Terry team strengths, critical hits in the simulation and re-tuning after the team effect gave no clear
gain. `experiments/log.csv` records every CV run of pipeline A.
