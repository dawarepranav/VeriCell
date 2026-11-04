"""Benchmark scoring follows MorphoOrgaVQA's published rules."""

import numpy as np

from vericell.vqa_score import groundtruth, score, score_field


def test_credit_thresholds():
    assert score_field(110.0, 100.0)["credit"] == 1.0   # 10% error
    assert score_field(120.0, 100.0)["credit"] == 0.5   # 20% error
    assert score_field(140.0, 100.0)["credit"] == 0.0   # 40% error
    assert score_field(None, 100.0)["credit"] == 0.0    # missing answer


def test_absolute_fallback_near_zero():
    r = score_field(0.0005, 0.0)
    assert r["status"] == "abs" and r["credit"] == 1.0


def test_groundtruth_has_every_slot(make_disk):
    lab = np.zeros((200, 200), np.int32)
    lab[make_disk(20, cx=50, cy=50)] = 1
    lab[make_disk(30, cx=130, cy=130)] = 2
    gt = groundtruth(lab)
    for c in ("Area", "Perimeter", "Roundness", "Roughness"):
        assert gt[f"{c}__max"] and gt[f"{c}__tertile_mean"]
    assert set(gt["Area__max"]) == {"Area_outer", "x", "y"}


def test_perfect_predictions_score_full_credit(make_disk):
    lab = np.zeros((200, 200), np.int32)
    lab[make_disk(25)] = 1
    gts = {"img.png": groundtruth(lab)}
    preds = {"img.png": {slot: {"clear": {"final_answer": ans}, "open": {"final_answer": ans}}
                         for slot, ans in gts["img.png"].items()}}
    recs = score(preds, gts)
    assert recs and all(r["credit"] == 1.0 for r in recs)
