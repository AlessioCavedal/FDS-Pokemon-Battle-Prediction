# Pokémon Battle Prediction – FDS Kaggle Challenge 2026

Predicting the winner of Gen 2 OU Pokémon battles from Player 1's full team, Player 2's lead and the first
30 turns of the battle, with classical machine learning only.

## Approach

1. **Timeline semantics.** Every timeline entry is a decision point of Player 1: the state at the end of the
   turn, plus an extra entry when a Player 1 Pokémon faints. The move field stores the *last* move used by the
   active Pokémon, so fresh moves are detected as changes of that field.
2. **Hidden faints.** Player 2 faints are only recorded when they coincide with a Player 1 faint; otherwise the
   fainted Pokémon silently disappears. A LightGBM detector is trained on Player 1 transitions, where faints are
   known (the faint entry is removed so that they look like Player 2's), and applied, cross-fitted, to every
   Player 2 disappearance (AUC ≈ 0.98, about one hidden faint per battle). Its probabilities drive all the
   Player 2 state estimates.
3. **Features.** Mostly Player 1 minus Player 2, so that both players share the same weights:
   - battle state at the last decision point (faints, HP, status) and its trajectory at earlier snapshots;
   - the two active Pokémon: boosts, volatile effects, speed order, Gen 2 damage and turns-to-KO estimates;
   - a matchup matrix between the remaining Pokémon of both teams, and win conditions (hazards, spinners,
     setup, phazers, healers, sleepers);
   - a quick simulation of the rest of the battle as a sequence of one-on-one duels between the remaining
     Pokémon (Gen 2 damage with the revealed moves, alone or with the typical moves of each species learned
     from the training battles; the winner keeps its remaining HP and the loser sends in its best answer),
     and the duel between the two actives;
   - action statistics (move classes, attack effectiveness, switches, damage dealt, healing) and switching
     behaviour: each player's chance to switch out in the current matchup and its switches beyond the
     expected ones, from voluntary-switch rates per pair of actives learned from the training battles;
   - static team descriptors and Player 2's lead;
   - symmetric bags of species, alive HP and revealed moves.
4. **Team effect.** Teams are reused across battles: there are 4,167 distinct Player 1 teams in the 10,000
   training battles, and 70% of the test battles use a Player 1 team that also appears in training. A rare team
   is a good proxy for the player behind it. For every team, the team effect is the shrunk mean residual
   (outcome minus the cross-fitted logistic-regression probability) of the training battles played with it:
   it counts positively when the team was Player 1's and negatively when it was Player 2's. Player 2's team is
   identified when all six Pokémon are revealed, or when the five revealed ones belong to a single known team.
   A head-to-head term does the same for battles between the same two teams, for example series between the
   same players. The encoding is cross-fitted, and the team-matching map is rebuilt from the training rows of
   each fold.
5. **Models.** A logistic regression with splines on the key state features, plus tuned LightGBM, XGBoost and
   CatBoost models. The tree models receive the bag features compressed into a cross-fitted
   logistic-regression logit, a cross-fitted logit of a logistic regression on all features, and the
   team-effect features. Blends average the models' logits with non-negative weights fitted on out-of-fold
   predictions.

Only the competition data and the supplied type chart are used. The Pokédex (types and base stats) is built
from the training battles, and the test set is only used for prediction.

## Validation

- Repeated stratified group K-fold, 5 folds × 3 repeats; exact duplicate timelines share a group.
- Accuracy is the target metric, with log-loss and AUC as tie-breakers. Changes are compared fold by fold on
  identical splits.
- Every target-dependent step is fitted inside the training folds: the hidden-faint detector, the stacked
  logits and the team effect are cross-fitted, and blend weights are evaluated with a nested split.
- Hyper-parameters are tuned with Optuna on a different fold split than the one used to compare models.

## Main results

5×3 cross-validation on the training set (`experiments/log.csv`):

| Step | Pipeline | Accuracy | Log-loss |
|---|---|---|---|
| Logistic regression, first feature set | `lr_base` | 0.6932 | 0.5736 |
| LightGBM with the stacked bag logit | `lgbm_stack` | 0.7031 | 0.5635 |
| + matchup matrix and win conditions | `lgbm_stack_v2` | 0.7047 | 0.5599 |
| Spline logistic regression | `lrs_mk_v4ref` | 0.7055 | 0.5571 |
| Tuned XGBoost with stacked logits | `xgb_t2` | 0.7081 | 0.5564 |
| + team effect | `xgb_team` | 0.7113 | 0.5521 |
| + head-to-head term | `xgb_h2h` | 0.7128 | 0.5518 |
| + battle simulation and switching behaviour | `xgb_sim` | 0.7146 | 0.5495 |
| Same features with CatBoost | `cat_sim` | 0.7155 | 0.5497 |
| Blend of `lr_spline_sim`, `xgb_sim`, `lgbm_sim`, `cat_sim` | `blend_sim` | 0.7156 (nested) | 0.5483 |

## Experiments

`experiments/log.csv` records every cross-validation run with its full configuration. `configs/pipelines.json`
keeps the main pipelines and one representative of each experiment family.

Drop-one-group ablations (`abl_minus_*`) show that every feature group contributes. These experiments did not
improve validation and are not used by the final pipelines:
- time-share and count variants of the bags;
- the expected composition of Player 2's unrevealed team;
- pairwise interactions in the logistic regression;
- separate stacked logits for each bag;
- a spline-based stacked logit;
- training-set augmentation with truncated battles;
- monotone constraints and seed bagging;
- re-tuning after the team effect;
- separate Player 1 / Player 2 team terms;
- a joint Bradley–Terry estimate of the team strengths.

## Layout

```
src/
  config.py      paths, seed, threads and CV settings
  data.py        loading, type chart, Pokédex built from the training battles
  mechanics.py   Gen 2 stat and damage formulas
  replay.py      timeline parsing, move events, transitions
  faint.py       hidden-faint detector
  features/      feature groups (state, active, events, matchup, simulation, switching, bags, team, ...)
                 and the cached build
  models.py      model factory, stacked logits, team effect, spline logistic regression
  pipelines.py   named pipelines from configs/pipelines.json
  cv.py          repeated cross-validation, out-of-fold predictions, experiment log
  ensemble.py    blend weights fitted on out-of-fold predictions
  tune.py        Optuna hyper-parameter search
  cli.py         command line entry point
configs/         pipelines, blends (members and weights), tuned parameters
experiments/     experiment log
```

## Usage

```bash
pip install -r requirements.txt
python -m src.cli features                      # build and cache the features
python -m src.cli cv xgb_sim                    # 5x3 cross-validation of a pipeline
python -m src.tune xgb_team --trials 60         # hyper-parameter search
python -m src.cli blend lr_spline_sim xgb_sim lgbm_sim cat_sim --save blend_sim
python -m src.cli submit --blend blend_sim      # fit on all training battles, write submissions/blend_sim.csv
python -m src.cli submit --pipeline cat_sim     # same for a single pipeline
```

- `blend` uses the out-of-fold predictions produced by `cv`; `submit` only needs the configurations.
- The data folder is found automatically: the local folder `fds-pokemon-battles-prediction-2026/` or the
  Kaggle input directory. It can be set with `FDS_DATA_DIR`.
- `FDS_N_JOBS` sets the number of threads (default 4).
