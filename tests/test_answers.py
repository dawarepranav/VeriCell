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
