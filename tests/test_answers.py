"""Answers and their calibrated ranges."""

import numpy as np

from vericell import answers

PLAN_MAX = {"metric": "Area_outer", "operation": "max", "fields": ["Area_outer"]}
PLAN_COUNT = {"metric": "count", "operation": "count", "fields": ["count"]}


def test_extreme_answers_need_a_confident_object():
    table = [{"Area_outer": 500.0}, {"Area_outer": 900.0}]
    p = np.array([0.95, 0.6])  # the bigger object is doubtful
    value, winner = answers.point(PLAN_MAX, table, p, [0, 1])
    assert winner == 0 and value["Area_outer"] == 500.0


def test_count_is_expected_number_of_real_objects():
    table = [{"Area_outer": 1.0}] * 4
    value, _ = answers.point(PLAN_COUNT, table, np.array([1.0, 1.0, 0.5, 0.5]), [0, 1, 2, 3])
    assert value["count"] == 3.0


def test_relative_range_and_clipping():
    calib = {"count:count:value": {"q": 2.0, "kind": "rel"}}
    r = answers.ranges(PLAN_COUNT, {"count": 10.0}, {}, calib)
    assert r["count"] == [0.0, 30.0]  # 10 +- 2*10, clipped at zero


def test_mc_range_narrows_with_less_doubt():
    calib = {"count:count:value": {"q": 1.0, "kind": "mc"}}
    wide = answers.ranges(PLAN_COUNT, {"count": 10.0}, {"count": 5.0}, calib)["count"]
    tight = answers.ranges(PLAN_COUNT, {"count": 10.0}, {"count": 1.0}, calib)["count"]
    assert (tight[1] - tight[0]) < (wide[1] - wide[0])


def test_stability_finds_a_doubtful_object_that_would_take_over():
    table = [{"Area_outer": 500.0}, {"Area_outer": 900.0}, {"Area_outer": 450.0}]
    p = np.array([0.95, 0.6, 0.95])
    st = answers.stability(PLAN_MAX, table, p, [0, 1, 2], winner=0)
    assert [c["id"] for c in st["challengers"]] == [1]  # confirmed real, #1 becomes the largest
    assert st["challengers"][0]["gap"] == 0.8
    assert st["successor"]["id"] == 2  # if #0 isn't real, #2 is next


def test_stability_is_only_for_extreme_questions():
    assert answers.stability(PLAN_COUNT, [{"Area_outer": 1.0}], np.array([1.0]), [0], winner=None) is None


def test_min_flip_finds_one_object_that_breaks_a_claim():
    table = [{"Area_outer": 500.0}, {"Area_outer": 900.0}, {"Area_outer": 450.0}]
    p = np.array([0.95, 0.6, 0.95])
    r = answers.min_flip(PLAN_MAX, table, p, [0, 1, 2], [1], "at_most", 600.0)
    assert r["holds_now"] and r["broken"]
    assert [(f["id"], f["to_real"]) for f in r["flips"]] == [(1, True)]
    assert r["value_after"] == 900.0 and r["likelihood"] == 0.6


def test_min_flip_reports_false_and_robust_claims():
    table = [{"Area_outer": 500.0}, {"Area_outer": 450.0}]
    p = np.array([0.95, 0.5])
    assert not answers.min_flip(PLAN_MAX, table, p, [0, 1], [1], "at_most", 400.0)["holds_now"]
    r = answers.min_flip(PLAN_MAX, table, p, [0, 1], [1], "at_most", 600.0)
    assert r["holds_now"] and not r["broken"]  # the only doubtful object is smaller: nothing breaks it
