import numpy as np
import pandas as pd

from ctrisk.ml.text import TextFeatures

rng = np.random.default_rng(0)
WORDS = ["cancer", "placebo", "insulin", "pediatric", "phase", "randomized", "pain", "vaccine", "heart", "kidney"]
DOCS = pd.Series([" ".join(rng.choice(WORDS, 6)) for _ in range(300)], index=range(1000, 1300))


def test_transform_keeps_index_and_shape():
    tf = TextFeatures(n_components=4, min_df=1).fit(DOCS)
    X = tf.transform(DOCS)
    assert X.shape == (300, 4) and list(X.columns) == tf.columns == ["txt_00", "txt_01", "txt_02", "txt_03"]
    assert X.index.equals(DOCS.index)


def test_components_never_exceed_vocabulary():
    tiny = pd.Series([" ".join(rng.choice(WORDS[:3], 5)) for _ in range(100)])   # <= 12 uni/bigrams
    tf = TextFeatures(n_components=64, min_df=1).fit(tiny)
    assert tf.transform(tiny).shape[1] < 12


def test_unseen_words_and_missing_text_are_fine():
    tf = TextFeatures(n_components=4, min_df=1).fit(DOCS)
    X = tf.transform(pd.Series(["completely novel tokens", None, ""]))
    assert X.shape == (3, 4) and np.isfinite(X.to_numpy()).all()


def test_deterministic():
    a = TextFeatures(n_components=4, min_df=1).fit(DOCS).transform(DOCS)
    b = TextFeatures(n_components=4, min_df=1).fit(DOCS).transform(DOCS)
    assert np.allclose(a, b)
