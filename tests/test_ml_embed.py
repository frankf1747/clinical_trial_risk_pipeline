import numpy as np
import pandas as pd

from ctrisk.ml.embed import EmbeddingFeatures, chunk_ids, pooled


class FakeTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(range(len(text.split())))}


def test_chunk_ids_split_long_texts_into_windows_and_cap_them():
    ids = list(range(1200))
    windows = chunk_ids(ids, window=510, max_chunks=2)
    assert [len(w) for w in windows] == [510, 510]                       # third window dropped by the cap
    assert chunk_ids([], 510, 4) == [[]]


def test_pooled_is_the_length_weighted_mean_renormalized():
    vecs = np.array([[1.0, 0.0], [0.0, 1.0]])
    out = pooled(vecs, lengths=[3, 1])
    assert np.allclose(out, np.array([0.75, 0.25]) / np.linalg.norm([0.75, 0.25]))


def test_embedding_features_reduce_on_training_rows_only():
    rng = np.random.default_rng(0)
    train = pd.Series(list(rng.normal(size=(200, 16)).astype("float32")))
    other = pd.Series(list(rng.normal(size=(50, 16)).astype("float32")) + [None])
    f = EmbeddingFeatures(n_components=4).fit(train)
    X = f.transform(other)
    assert list(X.columns) == ["emb_00", "emb_01", "emb_02", "emb_03"] and len(X) == 51
    assert X.iloc[-1].isna().all()                                        # no text: missing, not zero


def _with_embeddings(frame, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(len(frame), 16)).astype("float32")
    base[:, 0] += frame["n_countries"].to_numpy()                         # the embedding carries the signal too
    return frame.assign(embedding=list(base))


def test_risk_model_uses_embeddings_and_reasons_fold_them_into_registration_text():
    from test_ml_train import synthetic

    from ctrisk.ml.model import RiskModel, collapse_text
    from ctrisk.ml.text import TextFeatures
    f = _with_embeddings(synthetic(1500))
    params = {"n_estimators": 40, "num_leaves": 15, "verbose": -1}
    m = RiskModel(["phase", "faers_reports"], params, text=TextFeatures(min_df=1),
                  embed=EmbeddingFeatures(n_components=4)).fit(f, f["label"].fillna(0))
    assert any(n.startswith("emb_") for n in m.feature_names)
    contrib, names = collapse_text(m.contributions(f), m.feature_names)
    assert names[-1] == "registration_text" and not any(n.startswith(("emb_", "txt_")) for n in names)
    assert contrib.shape[1] == len(names)


def test_training_fits_embeddings_on_training_rows_and_reports_both_text_kinds(monkeypatch):
    from test_ml_train import SMALL_GRID, synthetic

    from ctrisk.ml.train import run
    seen = []
    original = EmbeddingFeatures.fit
    monkeypatch.setattr(EmbeddingFeatures, "fit", lambda self, e: seen.append(len(e)) or original(self, e))
    f = _with_embeddings(synthetic(2000))
    _, report = run(f, grid=SMALL_GRID, text_min_df=1, exclude=())
    n_train = int(((f["split"] == "train") & f["label"].notna()).sum())
    assert seen and max(seen) <= n_train                                  # never fit beyond the training rows
    models = report["targets"]["label"]["models"]
    assert {"lightgbm_tfidf_only", "lightgbm_embeddings_only"} <= set(models)
    assert "embedding" not in report["features"]["columns"]
