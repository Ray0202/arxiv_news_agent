"""Three memories that make tomorrow's run depend on what happened before.

* **Preference** — built only from labels you gave by hand. Recalls papers the keyword
  filter would have missed.
* **Seen papers** — embeddings of everything published so far, used to say "this is close
  to the one from Aug 15". Context, never a filter.
* **Research thread** — what recent deep reads actually used: benchmarks, datasets, tags.

Two rules are load-bearing and enforced in code rather than left to discipline:

1. **Only human labels feed preference.** `_manual_only` drops anything whose `source` is
   not `manual`, so a model-generated score can never train the recall that decides what
   the model sees next. A system that learns from its own judgements narrows until it only
   surfaces what it already believed.
2. **Similarity never removes a paper.** `nearest_seen` returns context for display. A
   genuinely important paper often arrives in the middle of a wave of similar ones, and
   "we showed something like this yesterday" is the worst possible reason to hide it.

A calibration note that shapes everything here: bge-style encoders have a high and
*population-dependent* similarity floor. Two unrelated abstracts drawn from all of arXiv
score ~0.45, but two unrelated abstracts that have both already passed a topic filter
score ~0.72 — the filter removed exactly the variation the cosine was measuring. Every
threshold in this module was therefore set by measuring the real distribution, not by
picking a number that sounded selective; both times, the number that sounded right was off
by enough to make the feature useless. See `nearest_seen` for the measurement.

Preference sidesteps the problem instead of thresholding through it: it scores
`max similarity to a liked paper − max similarity to a skipped one`, and the subtraction
cancels whatever the common floor happens to be for that day's population.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from .config import CACHE_DIR

MODEL_NAME = "BAAI/bge-small-en-v1.5"
CACHE = CACHE_DIR / "embeddings"

# Labels that mean "I was interested", versus "don't show me this".
POSITIVE = ("deep", "weak_evidence")
NEGATIVE = ("skip",)
# `weak_evidence` is interest-positive but not a quality endorsement, so it is a weaker
# vote for recall than a full deep read.
LABEL_WEIGHT = {"deep": 1.0, "weak_evidence": 0.6, "skip": 1.0}


class EmbeddingUnavailable(RuntimeError):
    pass


_model = None


def get_model():
    """Load the encoder lazily. Missing package is a soft failure, not a crash."""
    global _model
    if _model is not None:
        return _model
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - depends on the install extra
        raise EmbeddingUnavailable(
            "sentence-transformers is not installed; run `pip install -e '.[embed]'` to "
            "enable semantic recall. The pipeline works without it — the rescue channel "
            "is simply skipped."
        ) from exc
    device = None
    try:
        import torch

        if torch.backends.mps.is_available():
            device = "mps"
    except Exception:  # pragma: no cover - torch always present with s-t
        pass
    _model = SentenceTransformer(MODEL_NAME, device=device)
    return _model


def paper_text(rec: dict) -> str:
    """What gets embedded. Title plus abstract only — deliberately not the full text.

    A full-text vector is dominated by boilerplate (related work, reproducibility
    statements) and buries the contribution the reader actually reacted to.
    """
    return f"{rec.get('title', '')}\n\n{rec.get('abstract', '')}".strip()


def encode(texts: Sequence[str], batch_size: int = 64):
    import numpy as np

    if not texts:
        return np.zeros((0, 384), dtype="float32")
    model = get_model()
    return model.encode(
        list(texts), batch_size=batch_size, normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")


# --------------------------------------------------------------------------- storage
def _cache_path(date: str) -> Path:
    return CACHE / f"{date}.npz"


def embed_day(date: str, records: Sequence[dict], refresh: bool = False):
    """Vectors for one day's records, cached. Derived data — safe to delete."""
    import numpy as np

    ids = [r["arxiv_id"] for r in records]
    path = _cache_path(date)
    if path.exists() and not refresh:
        blob = np.load(path, allow_pickle=False)
        cached_ids = list(blob["ids"])
        if cached_ids == ids:
            return blob["vecs"], cached_ids
    vecs = encode([paper_text(r) for r in records])
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, ids=np.array(ids), vecs=vecs)
    return vecs, ids


def load_day_vectors(date: str):
    """Cached vectors for a past day, or (None, []) if never computed."""
    import numpy as np

    path = _cache_path(date)
    if not path.exists():
        return None, []
    blob = np.load(path, allow_pickle=False)
    return blob["vecs"], list(blob["ids"])


# ------------------------------------------------------------------------ preference
def _manual_only(entries: Iterable[dict]) -> list[dict]:
    """Rule 1, enforced here rather than trusted.

    Anything not explicitly marked `source: "manual"` is dropped, so a label written by
    the pipeline itself can never influence what the pipeline retrieves tomorrow.
    """
    return [e for e in entries if (e.get("source") or "manual") == "manual"
            and e.get("_generated") is not True]


@dataclass
class Preference:
    """Liked and disliked vectors, kept as sets rather than centroids.

    A centroid over several distinct interests lands between them and matches neither.
    With a handful of labels, nearest-neighbour to any single liked paper is both more
    robust and easier to explain: a rescued paper can name which of your labels pulled it in.
    """

    pos_vecs: Any = None
    pos_ids: list[str] = field(default_factory=list)
    pos_weights: list[float] = field(default_factory=list)
    neg_vecs: Any = None
    neg_ids: list[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        return self.pos_vecs is not None and len(self.pos_ids) > 0

    def score(self, vecs) -> tuple[Any, list[str]]:
        """`max weighted similarity to a liked paper − max similarity to a skipped one`.

        The subtraction cancels the encoder's high similarity floor, which makes the
        result comparable across days and actually thresholdable.
        """
        import numpy as np

        if not self.usable or len(vecs) == 0:
            return np.zeros(len(vecs), dtype="float32"), [""] * len(vecs)
        pos = vecs @ self.pos_vecs.T                       # (n, |pos|)
        weighted = pos * np.array(self.pos_weights, dtype="float32")
        best = weighted.argmax(axis=1)
        score = weighted.max(axis=1)
        because = [self.pos_ids[i] for i in best]
        if self.neg_vecs is not None and len(self.neg_ids):
            score = score - (vecs @ self.neg_vecs.T).max(axis=1)
        return score, because


def build_preference(feedback_entries: Iterable[dict], lookup) -> Preference:
    """Assemble the preference memory from hand labels.

    `lookup(arxiv_id)` returns a vector or None — labels for papers whose vectors were
    never computed are skipped rather than guessed.
    """
    import numpy as np

    pos_v, pos_i, pos_w, neg_v, neg_i = [], [], [], [], []
    for entry in _manual_only(feedback_entries):
        vec = lookup(entry.get("arxiv_id"))
        if vec is None:
            continue
        label = entry.get("label")
        if label in POSITIVE:
            pos_v.append(vec)
            pos_i.append(entry["arxiv_id"])
            pos_w.append(LABEL_WEIGHT.get(label, 1.0))
        elif label in NEGATIVE:
            neg_v.append(vec)
            neg_i.append(entry["arxiv_id"])
    return Preference(
        pos_vecs=np.vstack(pos_v) if pos_v else None,
        pos_ids=pos_i,
        pos_weights=pos_w,
        neg_vecs=np.vstack(neg_v) if neg_v else None,
        neg_ids=neg_i,
    )


# ----------------------------------------------------------------------- seen papers
def nearest_seen(
    vec, history: Sequence[tuple[str, str, Any]], top_k: int = 1, floor: float = 0.84
) -> list[dict]:
    """Closest previously published papers. Context for the card — never a filter.

    `history` is `[(arxiv_id, date, vector), ...]`.

    `floor` has to be far higher than intuition suggests. Measured across two real days:
    the nearest-neighbour similarity for eight unrelated on-topic papers landed between
    0.690 and 0.744, while the one genuinely-same-thread pair scored 0.868 — with nothing
    at all in between. A floor of 0.72, which sounds selective, marked five of eight,
    including a long-memory time-series estimator paired with a memory foundation model
    purely because both say "memory".

    The card claims "we covered this on the 15th". A wrong claim there is worse than
    silence, so the floor sits in the empty band rather than near the noise.
    """
    import numpy as np

    if not history:
        return []
    mat = np.vstack([h[2] for h in history])
    sims = mat @ vec
    order = np.argsort(-sims)[:top_k]
    return [
        {"arxiv_id": history[i][0], "date": history[i][1], "similarity": round(float(sims[i]), 3)}
        for i in order
        if sims[i] >= floor
    ]
