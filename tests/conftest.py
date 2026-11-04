import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def disk(r: float, size: int = 200, cx: float = 100, cy: float = 100) -> np.ndarray:
    yy, xx = np.mgrid[:size, :size]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r


@pytest.fixture
def make_disk():
    return disk
