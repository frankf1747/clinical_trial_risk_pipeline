"""Registration text -> a few dense columns: TF-IDF over word 1-2grams, compressed with SVD.

Fit on training trials only, so nothing about test trials shapes the vocabulary or components.
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


class TextFeatures:
    def __init__(self, n_components: int = 64, min_df: int = 20, max_features: int = 50_000):
        self.n_components, self.min_df, self.max_features = n_components, min_df, max_features
        self.tfidf = self.svd = None
        self.columns: list[str] = []

    def fit(self, texts: pd.Series) -> "TextFeatures":
        self.tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=self.min_df, max_features=self.max_features,
                                     sublinear_tf=True, dtype=np.float32)
        matrix = self.tfidf.fit_transform(texts.fillna(""))
        k = min(self.n_components, matrix.shape[1] - 1)
        self.svd = TruncatedSVD(n_components=k, random_state=0).fit(matrix)
        self.columns = [f"txt_{i:02d}" for i in range(k)]
        return self

    def transform(self, texts: pd.Series) -> pd.DataFrame:
        dense = self.svd.transform(self.tfidf.transform(texts.fillna("")))
        return pd.DataFrame(dense, columns=self.columns, index=texts.index)
