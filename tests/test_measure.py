"""The measurement engine: geometry must be right, because no language model checks it."""

import numpy as np
import pandas as pd

from vericell.measure import answer, drop_small, measure_label_mask, measure_object, tertile_means, tertile_means_soft


def test_disk_geometry(make_disk):
    m = measure_object(make_disk(40))
    assert abs(m["Area_outer"] - np.pi * 40 ** 2) / (np.pi * 40 ** 2) < 0.03
    assert abs(m["Perimeter_outer"] - 2 * np.pi * 40) / (2 * np.pi * 40) < 0.10
    assert 0.85 < m["roundness"] <= 1.05
    assert abs(m["x"] - 100) < 1 and abs(m["y"] - 100) < 1


def test_square_is_less_round_than_disk(make_disk):
    sq = np.zeros((200, 200), bool)
    sq[50:150, 50:150] = True
    assert measure_object(sq)["roundness"] < measure_object(make_disk(50))["roundness"]
    assert measure_object(sq)["perimeter_diff"] > measure_object(make_disk(50))["perimeter_diff"]


def test_label_mask_and_max_answer(make_disk):
    lab = np.zeros((200, 200), np.int32)
    lab[make_disk(20, cx=50, cy=50)] = 1
    lab[make_disk(35, cx=130, cy=130)] = 2
    table = measure_label_mask(lab)
    assert len(table) == 2
    big = answer(table, "Area_outer", "max", ["Area_outer", "x", "y"])
    assert abs(big["x"] - 130) < 1 and abs(big["y"] - 130) < 1


def test_tertiles_match_pandas_qcut():
    vals = np.array([5.0, 1.0, 9.0, 3.0, 7.0, 2.0, 8.0, 4.0, 6.0, 10.0, 11.0])
    ranks = pd.Series(vals).rank(method="first")
    tiers = pd.qcut(ranks, 3, labels=["low", "mid", "high"])
    expected = {t: float(pd.Series(vals)[tiers == t].mean()) for t in ("low", "mid", "high")}
    assert tertile_means(vals) == expected


def test_soft_tertiles_equal_hard_when_all_weights_equal():
    vals = np.arange(1, 10, dtype=float)
    soft, hard = tertile_means_soft(vals, np.ones(9)), tertile_means(vals)
    assert all(abs(soft[k] - hard[k]) < 1e-9 for k in hard)


def test_soft_tertiles_shift_when_small_objects_are_doubtful():
    vals = np.arange(1, 10, dtype=float)
    soft = tertile_means_soft(vals, np.r_[0.2, 0.2, 0.2, np.ones(6)])
    assert soft["low"] > tertile_means(vals)["low"]


def test_drop_small_removes_debris(make_disk):
    lab = np.zeros((200, 200), np.int32)
    lab[make_disk(20)] = 1
    lab[5:8, 5:8] = 2  # 9 px speck
    out = drop_small(lab, 200)
    assert set(np.unique(out)) == {0, 1}
