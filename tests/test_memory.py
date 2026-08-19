"""Preference memory: what may feed it, and what it is allowed to do."""

from __future__ import annotations

import numpy as np
import pytest

from pna import memory as mem


def _v(*xs):
    a = np.array(xs, dtype="float32")
    return a / np.linalg.norm(a)


VEC = {
    "liked_ts":    _v(1, 0, 0),
    "liked_agent": _v(0, 1, 0),
    "skipped":     _v(0, 0, 1),
    "new_ts":      _v(0.95, 0.05, 0.0),
    "new_chem":    _v(0.1, 0.1, 0.95),
}


def _lookup(aid):
    return VEC.get(aid)


def _label(aid, label, source="manual", **extra):
    return {"arxiv_id": aid, "label": label, "source": source, **extra}


# ------------------------------------------------------- rule 1: provenance is a gate
def test_only_manually_sourced_labels_reach_the_preference_memory():
    """A model-generated label must never train the retrieval that feeds the model."""
    entries = [
        _label("liked_ts", "deep"),
        _label("liked_agent", "deep", source="model_scored"),
        _label("skipped", "skip", source="auto"),
    ]
    pref = mem.build_preference(entries, _lookup)
    assert pref.pos_ids == ["liked_ts"]
    assert pref.neg_ids == [], "an auto-sourced negative is dropped too"


def test_a_generated_flag_also_excludes_an_entry():
    entries = [_label("liked_ts", "deep", _generated=True), _label("liked_agent", "deep")]
    assert mem.build_preference(entries, _lookup).pos_ids == ["liked_agent"]


def test_weak_evidence_counts_as_interest_but_weighs_less_than_a_deep_read():
    assert "weak_evidence" in mem.POSITIVE
    assert mem.LABEL_WEIGHT["weak_evidence"] < mem.LABEL_WEIGHT["deep"]


def test_labels_without_a_vector_are_skipped_not_guessed():
    pref = mem.build_preference([_label("never_embedded", "deep")], _lookup)
    assert not pref.usable


# ------------------------------------------------------------------------ scoring
def test_score_subtracts_the_negative_to_cancel_the_similarity_floor():
    """bge-style encoders put unrelated abstracts at ~0.45, so a raw cosine threshold
    passes everything. The difference is what makes the number thresholdable."""
    pref = mem.build_preference(
        [_label("liked_ts", "deep"), _label("skipped", "skip")], _lookup
    )
    scores, because = pref.score(np.vstack([VEC["new_ts"], VEC["new_chem"]]))
    assert scores[0] > 0.5, "close to a liked paper, far from the skipped one"
    assert scores[1] < 0, "close to the skipped one"
    assert because[0] == "liked_ts"


def test_score_names_which_label_pulled_a_paper_in():
    pref = mem.build_preference(
        [_label("liked_ts", "deep"), _label("liked_agent", "deep")], _lookup
    )
    _, because = pref.score(np.vstack([VEC["new_ts"], VEC["liked_agent"]]))
    assert because == ["liked_ts", "liked_agent"]


def test_preference_keeps_several_interests_apart_instead_of_averaging_them():
    """A centroid over time-series and agents lands between them and matches neither."""
    pref = mem.build_preference(
        [_label("liked_ts", "deep"), _label("liked_agent", "deep")], _lookup
    )
    scores, _ = pref.score(np.vstack([VEC["new_ts"]]))
    centroid = (VEC["liked_ts"] + VEC["liked_agent"]) / 2
    centroid = centroid / np.linalg.norm(centroid)
    assert scores[0] > float(VEC["new_ts"] @ centroid)


def test_an_empty_preference_scores_everything_zero_rather_than_failing():
    pref = mem.build_preference([], _lookup)
    scores, because = pref.score(np.vstack([VEC["new_ts"]]))
    assert list(scores) == [0.0] and because == [""]


# ----------------------------------------------- rule 2: similarity never removes
def test_nearest_seen_returns_context_and_respects_the_floor():
    history = [("old1", "2026-08-15", VEC["liked_ts"]), ("old2", "2026-08-16", VEC["skipped"])]
    near = mem.nearest_seen(VEC["new_ts"], history, top_k=2, floor=0.72)
    assert [n["arxiv_id"] for n in near] == ["old1"], "the unrelated one is below the floor"
    assert near[0]["date"] == "2026-08-15"
    assert 0.9 < near[0]["similarity"] <= 1.0


def test_default_floor_rejects_the_measured_noise_band():
    """Locks a calibration that two real days established and intuition got wrong.

    Nearest-neighbour similarity between unrelated papers that have *both* already passed
    the topic filter measured 0.690-0.744; the one real same-thread pair scored 0.868. The
    default must reject the former, or the card claims "we covered this" about a paper
    sharing nothing but a word.
    """
    import numpy as np

    def at(sim: float):
        # A vector whose cosine against e0 is exactly `sim`.
        v = np.zeros(384, dtype="float32")
        v[0], v[1] = sim, float(np.sqrt(1 - sim**2))
        return v

    probe = np.zeros(384, dtype="float32")
    probe[0] = 1.0
    noise = [(f"n{i}", "2026-07-30", at(s)) for i, s in enumerate((0.690, 0.729, 0.744))]
    assert mem.nearest_seen(probe, noise) == []

    real = noise + [("same-thread", "2026-07-30", at(0.868))]
    assert [n["arxiv_id"] for n in mem.nearest_seen(probe, real)] == ["same-thread"]


def test_nearest_seen_is_empty_rather_than_wrong_when_there_is_no_history():
    assert mem.nearest_seen(VEC["new_ts"], [], top_k=2) == []


def test_nearest_seen_has_no_way_to_reject_a_paper():
    """It returns annotations. There is no boolean, no threshold that means 'drop'."""
    out = mem.nearest_seen(VEC["new_ts"], [("a", "2026-08-15", VEC["new_ts"])], floor=0.0)
    assert isinstance(out, list)
    assert set(out[0]) == {"arxiv_id", "date", "similarity"}


def test_paper_text_is_title_and_abstract_only():
    """Embedding the full text buries the contribution under boilerplate."""
    text = mem.paper_text({"title": "T", "abstract": "A", "summary": {"article_zh": "X"}})
    assert text == "T\n\nA"
