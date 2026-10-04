# Experiment (M9): biomedical text embeddings vs TF-IDF

**Question.** The registration text is the model's strongest input, represented as TF-IDF word 1–2-grams compressed to 64 SVD components. Do biomedical sentence embeddings, which capture meaning rather than wording, predict termination better?

**Answer: no.** Embeddings alone are slightly worse than TF-IDF, and adding them to TF-IDF changes nothing. v4 stays the served model.

## Method

- **Encoder:** `NeuML/pubmedbert-base-embeddings` (PubMedBERT tuned for sentence similarity, 768 dimensions), pretrained and not fitted on our labels.
- **Long texts:** the median text is 600 tokens and 57% exceed the encoder's 512-token limit, so each text was cut into 510-token windows (up to 4, covering ~93% of texts whole), each window encoded, and the windows averaged by length. 111,118 trials in about 75 minutes on an Apple M1 Pro GPU (`ctrisk.ml.embed`).
- **Into the model:** PCA to 64 components fit on training rows only, the same budget as the TF-IDF SVD. Everything else as v4: the same frozen rows (`TRIAL_FEATURES_M9_EMBEDDINGS` equals v4's), splits, tuning grid, and the four leaky fields left out. The run trains TF-IDF + embeddings (the candidate), TF-IDF only, and embeddings only, all with the candidate's tuned parameters.

## Results (test: 10,242 trials that started 2015–2016)

| Text representation | Any termination AUC | Enrollment termination AUC | Top-10% precision |
|---|---|---|---|
| v4: TF-IDF (its own tuning) | 0.703 | 0.783 | 30.5% |
| TF-IDF + embeddings (candidate v5) | 0.708 (0.696–0.722) | 0.785 | 31.5% |
| TF-IDF only, same parameters | 0.709 | 0.783 | 30.4% |
| Embeddings only, same parameters | 0.699 | 0.783 | 30.0% |
| No text | 0.688 | 0.768 | 29.8% |

- Candidate minus v4, paired bootstrap on the same trials: any termination −0.002 to +0.011, enrollment −0.007 to +0.010. Not significant.
- The candidate's small edge over v4 comes from different tuned parameters, not from the embeddings: with the same parameters, TF-IDF alone scores 0.709 and adding embeddings gives 0.708.
- Rolling-origin windows tell the same story (candidate 0.716 / 0.729 / 0.701 / 0.708 vs v4 0.719 / 0.728 / 0.701 / 0.703).
- Embeddings carry most of the text signal on their own (no text 0.688 → embeddings 0.699 → TF-IDF 0.709), so they overlap with TF-IDF rather than add to it.

## Why, probably

The text signal that predicts termination looks lexical: specific terms (pilot, investigator-initiated, named rare conditions, eligibility phrasing) that TF-IDF keeps and a similarity-tuned encoder blurs into general topic. Averaging up to four windows and compressing to 64 components also discards detail. A model fine-tuned on the outcome could do better, but at a real risk of overfitting 36,000 training trials, and it would need the same point-in-time audit.

## What stays

- `ctrisk.ml.embed` and the `embed` dependency group stay, tested: the embeddings are a good base for a "similar past trials" search, which needs meaning rather than prediction.
- The candidate model and its metrics are kept in `models/experiments/m9-embeddings/` (not picked up by `make score`, `make lookup` or `make report`).
