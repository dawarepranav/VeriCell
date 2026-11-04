import numpy as np

from vericell.metrics import auroc, iou_matrix, match


def test_iou_identical_and_disjoint(make_disk):
    a, b = make_disk(20, cx=50, cy=50), make_disk(20, cx=150, cy=150)
    iou = iou_matrix([a, b], [a])
    assert iou[0, 0] == 1.0 and iou[1, 0] == 0.0


def test_one_to_one_matching(make_disk):
    a, b = make_disk(20, cx=50, cy=50), make_disk(20, cx=150, cy=150)
    _, pairs = match(iou_matrix([a, b], [b, a]))
    assert sorted(pairs) == [(0, 1), (1, 0)]


def test_auroc_bounds():
    assert auroc(np.array([0.9, 0.8, 0.1, 0.2]), np.array([1, 1, 0, 0])) == 1.0
    assert auroc(np.array([0.5, 0.5]), np.array([1, 0])) == 0.5
