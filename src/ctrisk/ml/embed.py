"""Registration text as biomedical sentence embeddings (M9), and their reduction for the model.

Encoder: NeuML/pubmedbert-base-embeddings (PubMedBERT tuned for sentence similarity, 768 dimensions),
pretrained and never fitted on our labels, so it cannot carry outcome information between splits. Texts run
past its 512-token limit for 57% of trials, which would cut the eligibility criteria; each text is cut into
510-token windows (up to 4, ~93% of texts whole), each window encoded, and the windows averaged by length.

    uv run --group embed python -m ctrisk.ml.embed            # all trials -> data/parquet/trial_embeddings/
    uv run --group embed python -m ctrisk.ml.embed --audit    # archived texts for the point-in-time audit

Shards of 5,000 trials are written as they finish, so an interrupted run resumes where it stopped.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

MODEL = "NeuML/pubmedbert-base-embeddings"
WINDOW, MAX_CHUNKS, SHARD = 510, 4, 5000


def chunk_ids(ids: list[int], window: int = WINDOW, max_chunks: int = MAX_CHUNKS) -> list[list[int]]:
    return [ids[i:i + window] for i in range(0, max(len(ids), 1), window)][:max_chunks] or [[]]


def pooled(vectors: np.ndarray, lengths: list[int]) -> np.ndarray:
    v = np.average(vectors, axis=0, weights=np.maximum(lengths, 1))
    return v / max(np.linalg.norm(v), 1e-12)


def encode(texts: list[str], model) -> np.ndarray:
    """One unit vector per text: the length-weighted mean of its window embeddings."""
    tok = model.tokenizer
    windows, owner, lengths = [], [], []
    for i, text in enumerate(texts):
        for ids in chunk_ids(tok(text or "", add_special_tokens=False)["input_ids"]):
            windows.append(tok.decode(ids))
            owner.append(i)
            lengths.append(len(ids))
    vecs = model.encode(windows, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
    owner, lengths = np.array(owner), np.array(lengths)
    return np.stack([pooled(vecs[owner == i], lengths[owner == i]) for i in range(len(texts))]).astype("float32")


class EmbeddingFeatures:
    """768-dim embeddings -> a few PCA components, fit on training rows only (like TextFeatures' SVD).
    Rows without an embedding come back as missing, which LightGBM handles natively."""

    def __init__(self, n_components: int = 64):
        self.n_components = n_components
        self.pca: PCA | None = None
        self.columns: list[str] = []

    def fit(self, embeddings: pd.Series) -> "EmbeddingFeatures":
        X = np.stack([e for e in embeddings if e is not None])
        self.pca = PCA(n_components=min(self.n_components, X.shape[1]), random_state=0).fit(X)
        self.columns = [f"emb_{i:02d}" for i in range(self.pca.n_components_)]
        return self

    def transform(self, embeddings: pd.Series) -> pd.DataFrame:
        has = embeddings.map(lambda e: e is not None and not (isinstance(e, float) and np.isnan(e)))
        out = pd.DataFrame(np.nan, index=embeddings.index, columns=self.columns)
        if has.any():
            out.loc[has] = self.pca.transform(np.stack(embeddings[has].to_list()))
        return out


def attach(frame: pd.DataFrame, path: str | Path) -> pd.DataFrame:
    """frame with an `embedding` column (array or None) joined by nct_id from the embeddings store."""
    emb = pd.read_parquet(path, columns=["nct_id", "embedding"])
    lookup = dict(zip(emb["nct_id"], emb["embedding"].map(lambda v: np.asarray(v, dtype="float32"))))
    return frame.assign(embedding=frame["nct_id"].map(lookup).where(frame["nct_id"].isin(lookup), None))


if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer

    from ctrisk.config import load_config

    cfg = load_config()
    audit = "--audit" in sys.argv
    if audit:                                    # the record each trial had at its start, for the audit
        texts = pd.read_parquet(Path(cfg.local_root, "parquet_pit", "registry_at_start"), columns=["nct_id", "text"])
        out = Path(cfg.local_root, "parquet_pit", "embeddings")
    else:
        texts = pd.read_parquet(Path(cfg.local_root, "parquet", "trial_text"), columns=["nct_id", "text"])
        out = Path(cfg.local_root, "parquet", "trial_embeddings")
    out.mkdir(parents=True, exist_ok=True)
    texts = texts.sort_values("nct_id").reset_index(drop=True)
    model = SentenceTransformer(MODEL, device="mps")
    for n, lo in enumerate(range(0, len(texts), SHARD)):
        shard = out / f"part-{n:03d}.parquet"
        if shard.exists():
            continue
        part = texts.iloc[lo:lo + SHARD]
        vecs = encode(part["text"].fillna("").tolist(), model)
        pd.DataFrame({"nct_id": part["nct_id"].to_numpy(), "embedding": list(vecs)}).to_parquet(shard)
        print(f"{shard.name}: {lo + len(part):,} of {len(texts):,}", flush=True)
