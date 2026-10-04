# Experiment (v6): stable scores for single trials

**Question.** Two fits of the same model on the same data disagreed about individual trials far more than their identical AUCs suggested: in the lookup, 30% of the cross-fitted 2008–2014 trials moved more than 10 percentile points between two builds that differed only in row order. Where does that noise come from, and can it be removed without losing accuracy?

**Answer: two sources, both fixed.** Half of the text components depended on row order, and LightGBM's row and column subsampling depends on the seed. With an exact SVD and an ensemble of 10 seeds, two independent refits put only 3% of trials more than 10 percentile points apart (38% with v5's setup), and accuracy is unchanged or better. Served as model v6 (and survival v4).

## Where the noise came from

1. **The text SVD was not converged.** scikit-learn's default randomized TruncatedSVD, fit on the same training texts in a different order, gave the same top components but different lower ones: 32 of 64 components correlated below 0.99 between orders, the worst at 0.06. Every downstream score moved with them. ARPACK computes the components exactly: all 64 identical across orders, signs included, in about 30 seconds per fit.
2. **LightGBM's subsampling.** Each tree sees 80% of rows and 80% of columns, drawn from the seed. One fit's score for one trial is one draw.

Measured on v5's frozen data and tuned settings: two replicas, each refit with other seeds and shuffled training rows, scoring the same 40,136 trials (test and active), percentiles among active trials as the lookup shows them.

| Ensemble size | Test AUC | Rank correlation between replicas | Trials moving > 10 percentile points |
|---|---|---|---|
| Randomized SVD, 1 fit | 0.707 | 0.904 | 38% |
| Randomized SVD, 10 fits | 0.711 | 0.962 | 22% |
| Exact SVD, 1 fit | 0.707 | 0.931 | 32% |
| Exact SVD, 3 fits | 0.708 | 0.975 | 15% |
| Exact SVD, 5 fits | 0.709 | 0.985 | 8% |
| Exact SVD, 10 fits | 0.710 | 0.992 | 2.4% |

With the randomized SVD, more seeds stopped helping, because the text noise remained. With the exact SVD, the typical change in a trial's score falls about as 1/√k, as averaging independent draws should: from 1.7 to 0.54 percentage points at 10 fits.

## v6

Every LightGBM fit in training (the scored models, the ablations and the rolling-origin windows) is now an ensemble of 10 seeds with log-odds averaged; tuning still uses single fits. SHAP contributions are averaged the same way, so they still add up to the ensemble's log-odds. Training reports the refit check itself (`refit_stability` in the metrics, a section in the model card).

| | v5 (one fit) | v6 (10 seeds, exact SVD) |
|---|---|---|
| Test AUC, any termination | 0.708 (0.696–0.721) | 0.710 (0.696–0.722) |
| Test AUC, enrollment termination | 0.780 | 0.789 (0.770–0.807) |
| Top-10% termination rate | 30.4% | 31.7% |
| Calibration slope / intercept | 0.94 / 0.12 | 1.08 / 0.11 |
| Point-in-time audit (18,568 trials, 2017–2020) | +0.003 (0.000 to +0.005) | +0.001 (−0.001 to +0.003) |
| Refit: trials moving > 10 percentile points (v5: from the table above) | 38% | 3.2% |
| Refit: rank correlation | 0.904 | 0.992 |

- v6 minus v5, paired bootstrap on the same 10,242 test trials (`python -m ctrisk.ml.compare 5 6`): any termination +0.002 (−0.003 to +0.006), enrollment +0.009 (+0.003 to +0.015).
- Rolling origin: 0.718, 0.731, 0.707, 0.710 (v5: 0.713, 0.731, 0.702, 0.708).
- The calibration slope rose above 1: an average of fits is less extreme than any one of them, so the scores are now slightly too cautious rather than slightly too extreme.

## Survival model

The lookup's other number, a running trial's chance of termination in the next 2 years, came from one survival fit and was noisier still. Survival v2 and v3, two fits that differed only in row order, agreed at Spearman 0.867 on the 26,251 running trials, and 43% moved more than 10 percentile points. Survival v4 uses the same exact SVD and 10 seeds, with each period's class log-odds averaged before the softmax.

| Next-2-year risk of running trials, two independent refits | One fit | Ensemble of 10 |
|---|---|---|
| Rank correlation | 0.896 | 0.989 |
| Median change in percentile | 7.2 points | 2.3 points |
| Trials moving > 10 percentile points | 38% | 3.5% |

Accuracy held or improved. Time-AUC for termination within 2 years: 0.649 (0.630–0.672) on 2015–16 starts and 0.653 (0.638–0.666) on 2017–20 starts. Gain over the yes/no model: +0.016 to +0.060 and +0.057 to +0.089. The 2015–16 gain is now significant. Single survival fits had put it at +0.019 to +0.026 with intervals crossing zero, but part of the change is the baseline: v6 ranks 2-year terminations slightly lower than v5 did (0.613 vs 0.622). At the five landmark dates, ranking the trials then running: 0.595–0.613 vs 0.536–0.559 for the yes/no model. Recalibration: terminating −0.81 + 0.70 × learned log-odds.

## Cost

Training takes about 10× longer in LightGBM time (v6 trained in 37 minutes on a laptop), and the lookup's cross-fitting, 7 folds × 2 targets, about 10× too. Scoring and serving are unchanged: the lookup reads precomputed scores.
