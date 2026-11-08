"""Tests for the binding-protocol split restriction (course spec 2026-11-08:
temperature/Platt fits use ONLY the 2,896 calib IDs of the held-out split)."""

from pathlib import Path

import numpy as np
import pytest

from qa_baseline.calibrate import _filter_to_qids, _split_qids


def _write_split(tmp_path, parts):
    path = tmp_path / "split.json"
    path.write_text(
        json.dumps({"seed": 42, "parts": parts}), encoding="utf-8"
    )
    return path


import json


def test_split_qids_reads_parts(tmp_path):
    path = _write_split(tmp_path, {
        "head_train": {"ids": ["a", "b"]},
        "calib": {"ids": ["c", "d", "e"]},
    })
    assert _split_qids(path, "calib") == {"c", "d", "e"}
    assert _split_qids(path, "head_train") == {"a", "b"}


def test_split_qids_unknown_part(tmp_path):
    path = _write_split(tmp_path, {"calib": {"ids": ["c"]}})
    with pytest.raises(ValueError, match="not in split file"):
        _split_qids(path, "test")


def test_filter_to_qids_keeps_order_and_values():
    scores = {
        "qid": np.array(["a", "c", "b", "e"]),
        "s": np.arange(8, dtype=np.float64).reshape(4, 2),
        "e": np.arange(8, dtype=np.float64).reshape(4, 2) + 1,
        "mask": np.ones((4, 2), dtype=bool),
    }
    y = np.array([1.0, 0.0, 1.0, 1.0])
    fs, fy, keep = _filter_to_qids(scores, y, {"c", "a", "e"})
    assert fs["qid"].tolist() == ["a", "c", "e"]
    assert fy.tolist() == [1.0, 0.0, 1.0]
    assert keep.tolist() == [True, True, False, True]
    # arrays filtered consistently
    assert fs["s"][0, 0] == 0.0 and fs["s"][1, 0] == 2.0


def test_filter_to_qids_missing_id_raises():
    scores = {"qid": np.array(["a", "b"])}
    with pytest.raises(ValueError, match="disagree"):
        _filter_to_qids(scores, np.zeros(2), {"a", "zzz"})


def test_real_split_calib_is_2896_and_disjoint():
    """The binding protocol fixes the calib part at 2,896 IDs (Ziying's
    splits/heldout-v1.json, commit f24d049). Guard against regeneration drift."""
    path = Path("splits/heldout-v1.json")
    if not path.exists():
        pytest.skip("splits/heldout-v1.json not present")
    doc = json.loads(path.read_text(encoding="utf-8"))
    parts = doc["parts"]
    assert set(parts) == {"head_train", "calib", "test"}
    assert len(parts["calib"]["ids"]) == 2896
    assert len(parts["head_train"]["ids"]) + len(parts["test"]["ids"]) == 5845
    ids = [set(p["ids"]) for p in parts.values()]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
