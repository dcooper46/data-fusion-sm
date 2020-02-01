from operator import ne

import numpy as np
from numpy import testing as tst
import pandas as pd
import pytest

from datafusionsm.implicit_model import HotDeck


donors = pd.DataFrame([
    [0, 1, 3, 2, 0],
    [1, 0, 1, 1, 1],
    [2, 1, 2, 2, 1],
    [3, 1, 3, 1, 0]], columns=["id", "a", "b", "c", "target"]
).astype('category')

recipients = pd.DataFrame([
    [0, 1, 3, 2],
    [1, 0, 1, 1],
    [2, 0, 1, 2],
    [3, 0, 3, 2]], columns=["id", "a", "b", "c"]
).astype('category')


def test_default_setup():
    hd = HotDeck()

    tst.assert_equal(hd.match_method, "nearest")
    tst.assert_equal(hd.score_method, "cosine")
    assert(hd.minimize is True)
    assert(hd.importance is None)

    hd.fit(donors, recipients)

    tst.assert_array_equal(hd.critical, [])
    tst.assert_equal(len(hd.linking), 6)
    tst.assert_equal(hd.imp_wgts, np.ones(len(hd.linking)))

    fused = hd.transform(donors, recipients)
    tst.assert_equal(len(fused), len(recipients))


def test_default_fit_transform():
    hd = HotDeck()
    fused = hd.fit_transform(donors, recipients)
    tst.assert_equal(len(fused), len(recipients))
    tst.assert_equal(fused.shape[1], len(donors.columns) + len(recipients.columns))


def test_fit_transform_specific_target():
    hd = HotDeck()
    fused = hd.fit_transform(donors, recipients, target="target")
    tst.assert_equal(len(fused), len(recipients))
    tst.assert_equal(fused.shape[1], len(recipients.columns) + 1)


def test_configurable_scoring_methods():
    for scorer in ["cosine", "euclidean", "manhattan"]:
        hd = HotDeck(score_method=scorer)
        hd.fit(donors, recipients)
        assert hd.matches is not None
        assert isinstance(hd.matches, pd.DataFrame)


def test_configurable_matching_methods():
    for matcher in ["nearest", "neighbors", "hungarian", "jonker_volgenant"]:
        hd = HotDeck(match_method=matcher)
        hd.fit(donors, recipients)
        assert hd.matches is not None
        assert isinstance(hd.matches, pd.DataFrame)


def test_configurable_importance_methods():
    for imp in ["entropy", "gini", "info_gain", "linear", "tree"]:
        hd = HotDeck(importance=imp)
        hd.fit(donors, recipients)
        assert hd.matches is not None
        assert isinstance(hd.matches, pd.DataFrame)
        tst.assert_array_compare(ne, hd.imp_wgts, np.ones(len(hd.imp_wgts)))


def test_configurable_scoring_args():
    hd = HotDeck(score_method="euclidean")
    hd.fit(donors, recipients, score_args={"squared": True})
    assert hd.matches is not None
    assert isinstance(hd.matches, pd.DataFrame)


def test_configurable_matching_args():
    hd = HotDeck(match_method="neighbors")
    hd.fit(donors, recipients, match_args={"k": 7, "flatness": 5})
    assert hd.matches is not None
    assert isinstance(hd.matches, pd.DataFrame)
