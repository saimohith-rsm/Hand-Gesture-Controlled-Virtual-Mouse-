"""test_filters.py
Tests for OneEuroFilter smoothing across consecutive frames, reset(), and real dt.
"""
from filters import OneEuroFilter

def test_one_euro_filter():
    f = OneEuroFilter(freq=30.0, min_cutoff=1.0, beta=0.05, deadzone=1.0)
    # Frame 1: initial point
    p1 = f((100.0, 100.0))
    assert p1 == (100.0, 100.0)

    # Frame 2: second point with dt=1/30
    p2 = f((110.0, 105.0), dt=1.0/30.0)
    assert isinstance(p2, tuple)
    assert len(p2) == 2

    # Frame 3: third point
    p3 = f((120.0, 110.0), dt=1.0/30.0)
    assert isinstance(p3, tuple)
    assert 100.0 < p3[0] < 125.0
    assert 100.0 < p3[1] < 115.0

def test_one_euro_filter_reset():
    f = OneEuroFilter(freq=30.0, min_cutoff=1.0, beta=0.05)
    f((100.0, 100.0))
    f((110.0, 105.0))
    assert f.x_prev is not None

    # Reset clears internal state
    f.reset()
    assert f.x_prev is None
    assert f.dx_prev == (0.0, 0.0)

    # Reappearance anchors directly to new position without drag from old position
    p_new = f((500.0, 500.0))
    assert p_new == (500.0, 500.0)
