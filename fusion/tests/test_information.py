"""
test the information theory functions
"""
import numpy as np
from fusion.utils import information as it


x = np.array([1, 1, 1, 2, 3, 3, 4, 5, 5, 5])
y = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 1])


def test_gini():
    assert np.isclose(it.gini(x), 0.76)
    assert np.isclose(it.gini(y), 0.48)


def test_entropy():
    assert np.isclose(it.entropy(x), 1.5048, rtol=1e-4)
    assert np.isclose(it.entropy(x, base=2), 2.1710, rtol=1e-4)
    assert np.isclose(it.entropy(y), 0.6730, rtol=1e-4)
    assert np.isclose(it.entropy(y, base=2), 0.9710, rtol=1e-4)


def test_efficiency():
    assert np.isclose(it.efficiency(x), 0.935, rtol=1e-4)
    assert np.isclose(it.efficiency(x, base=2), 1.3489, rtol=1e-4)
    assert np.isclose(it.efficiency(y), 0.971, rtol=1e-4)
    assert np.isclose(it.efficiency(y, base=2), 1.4008, rtol=1e-4)
