"""
test functions that calculate importance weights
"""
import numpy as np
import pandas as pd

from fusion.implicit import importance_weights as iw


y1 = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 1])
y2 = np.array([10, 21, 12, 43, 34, 15, 12, 33, 66, 27])
wgts = pd.Series(data=[0.25, 0.67, 0.88, 0.33], index=["a", "b", "c", "d"])
scaled_wgts = pd.Series(
    data=[4, 1.493, 1.1363, 3.03],
    index=["a", "b", "c", "d"]
)


def test_define_task():
    assert iw._define_task(y1) == "classification"
    assert iw._define_task(y2) == "regression"


def test_scale_wgts():
    iw_scaled_wgts = iw._scale_wgts(wgts)
    assert np.isclose(iw_scaled_wgts, scaled_wgts)
    assert iw_scaled_wgts.index == scaled_wgts.index
