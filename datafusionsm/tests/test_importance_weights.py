"""
test functions that calculate importance weights
"""
import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal, assert_index_equal

from datafusionsm.implicit_model import importance_weights as iw


y1 = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 1])
y2 = np.array([10, 21, 12, 43, 34, 15, 12, 33, 66, 27])
wgts = pd.Series(data=[0.25, 0.67, 0.88, 0.33], index=["a", "b", "c", "d"])
scaled_wgts = pd.Series(
    data=[4, 1.493, 1.136, 3.03],
    index=["a", "b", "c", "d"]
)

donors = pd.DataFrame(25 * [
    [1, 3, 2, 0],
    [0, 1, 1, 1],
    [1, 2, 2, 1],
    [0, 3, 1, 0]], columns=["a", "b", "c", "target"]
).astype('category')


def test_scale_wgts():
    iw_scaled_wgts = iw._scale_wgts(wgts).round(3)
    assert_series_equal(iw_scaled_wgts, scaled_wgts)
    assert_index_equal(iw_scaled_wgts.index, scaled_wgts.index)


def test_supervised_wgts():
    for method in ['info_gain', 'mutual_information', 'linear', 'tree']:
        weights = iw.supervised_imp_wgts(donors, "target", method)
        assert_index_equal(weights.index, pd.Index(["a", "b", "c"]))
        assert np.any(weights)  # not all 0 weights


def test_unsupervised_wgts():
    for method in ['entropy', 'efficiency', 'gini']:
        weights = iw.unsupervised_imp_wgts(
            donors.drop("target", axis=1), method)
        assert_index_equal(weights.index, pd.Index(["a", "b", "c"]))
        assert np.any(weights)  # not all 0 weights
