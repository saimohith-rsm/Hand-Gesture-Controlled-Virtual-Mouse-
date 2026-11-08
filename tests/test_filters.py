"""test_filters.py
Tests for OneEuroFilter smoothing across consecutive frames.
"""
from filters import OneEuroFilter

def test_one_euro_filter():
    f = OneEuroFilter(freq=30.0, min_cutoff=1.0, beta=0.05)
    # Frame 1: initial point
    p1 = f((100.0, 100.0))
    assert p1 == (100.0, 100.0)

    # Frame 2: second point (triggers derivative computation and dx_prev as tuple)
    p2 = f((110.0, 105.0))
    assert isinstance(p2, tuple)
    assert len(p2) == 2

    # Frame 3: third point (ensures no TypeError with tuple math)
    p3 = f((120.0, 110.0))
    assert isinstance(p3, tuple)
    assert 100.0 < p3[0] < 125.0
    assert 100.0 < p3[1] < 115.0

