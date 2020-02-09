from numpy import testing as tst
import numpy as np
import pandas as pd
from sklearn.svm import SVC

from datafusionsm.implicit_model import PIM
from datafusionsm.implicit_model.predictive_isotonic_matching import unconstrained_iso
from datafusionsm.utils.formatting import encode_columns


donors = pd.DataFrame(25 * [
    [0, 1, 3, 2, 0],
    [1, 0, 1, 1, 1],
    [2, 1, 2, 2, 1],
    [3, 0, 3, 1, 0]], columns=["id", "a", "b", "c", "target"]
).astype('category')

recipients = pd.DataFrame(25 * [
    [0, 1, 3, 2],
    [1, 0, 1, 1],
    [2, 0, 1, 2],
    [3, 0, 3, 2]], columns=["id", "a", "b", "c"]
).astype('category')


encoded_donors, d_ecols = encode_columns(donors.drop(["id", "target"], axis=1))
_, r_ecols = encode_columns(recipients.drop("id", axis=1))
linking = list(set(d_ecols).intersection(r_ecols))


def test_unconstrained_iso():
    n_bins = 4
    x, y = np.random.rand(10), np.random.rand(10)
    x_ids = np.array(list(enumerate(x)))
    y_ids = np.array(list(enumerate(y)))
    steps = np.arange(0, 1, round(1 / n_bins, 2))
    x_quants = [np.quantile(x, q) for q in steps]
    y_quants = [np.quantile(y, q) for q in steps]
    x_binned = np.digitize(x, x_quants, right=True)
    y_binned = np.digitize(y, y_quants, right=True)

    matches, scores = unconstrained_iso(x_ids, y_ids, x_binned, y_binned, n_bins)
    assert len(matches) == x.size
    assert len(scores) == x.size


def test_default_setup():
    pim = PIM("target")

    tst.assert_equal(pim.match_method, "unconstrained")
    tst.assert_equal(pim.model_method, None)

    pim.fit(donors, recipients)

    tst.assert_equal(pim.model_method, "linear")
    tst.assert_array_equal(pim.critical, [])
    tst.assert_equal(len(pim.linking), 6)
    tst.assert_equal(list(pim.results)[0], "target")
    tst.assert_equal(len(recipients), len(pim.results["target"]["matches"]))

    fused = pim.transform(donors, recipients)
    tst.assert_equal(len(fused), len(recipients))
    assert "target" in fused.columns


def test_default_fit_transform():
    pim = PIM("target")
    fused = pim.fit_transform(donors, recipients)
    tst.assert_equal(len(fused), len(recipients))
    tst.assert_equal(fused.shape[1], len(recipients.columns) + 1)


def test_critical_cells():
    pim = PIM("target")
    pim.fit(donors, recipients, critical="a")
    # print(pim.results["target"]["matches"].shape)
    tst.assert_array_equal(sorted(pim.critical), ["a_0", "a_1"])
    fused = pim.transform(donors, recipients)

    assert isinstance(pim.results["target"]["matches"], pd.DataFrame)
    tst.assert_equal(len(fused), len(recipients))
    assert "target" in fused.columns


def test_critical_cells_with_1_kfold():
    pim = PIM("target")
    model_args = {"k": 1}
    pim.fit(donors, recipients, critical="a", model_args=model_args)
    # print(pim.results["target"]["matches"].shape)
    tst.assert_array_equal(sorted(pim.critical), ["a_0", "a_1"])
    fused = pim.transform(donors, recipients)

    assert isinstance(pim.results["target"]["matches"], pd.DataFrame)
    tst.assert_equal(len(fused), len(recipients))
    assert "target" in fused.columns


def test_custom_model_probability_output():
    svc = SVC(probability=True)
    svc.fit(encoded_donors[linking], donors["target"])

    pim = PIM("target")
    model_args = {"model": svc}
    pim.fit(donors, recipients, model_args=model_args)
    fused = pim.transform(donors, recipients)

    tst.assert_equal(len(fused), len(recipients))

    # model trained within critical cells
    svc.fit(encoded_donors[linking].drop(["a_1", "a_0"], axis=1), donors["target"])
    pim.fit(donors, recipients, critical="a", model_args=model_args)
    fused = pim.transform(donors, recipients)

    assert isinstance(pim.results["target"]["matches"], pd.DataFrame)
    tst.assert_equal(len(fused), len(recipients))
    assert "target" in fused.columns


def test_custom_model_regression_output():
    svc = SVC()
    svc.fit(encoded_donors[linking], donors["target"])

    pim = PIM("target")
    model_args = {"model": svc}
    pim.fit(donors, recipients, model_args=model_args)
    fused = pim.transform(donors, recipients)

    # model trained within critical cells
    svc.fit(encoded_donors[linking].drop(
        ["a_1", "a_0"], axis=1), donors["target"])
    pim.fit(donors, recipients, critical="a", model_args=model_args)
    fused = pim.transform(donors, recipients)

    assert isinstance(pim.results["target"]["matches"], pd.DataFrame)
    tst.assert_equal(len(fused), len(recipients))
    assert "target" in fused.columns


def test_configurable_matching_methods():
    for matcher in ["unconstrained", "constrained"]:
        pim = PIM("target", match_method=matcher)
        if matcher == "constrained":
            with tst.assert_raises(NotImplementedError):
                pim.fit(donors, recipients)
        else:
            pim.fit(donors, recipients)
            assert isinstance(pim.results["target"]["matches"], pd.DataFrame)


def test_configurable_matching_args():
    pim = PIM("target")
    pim.fit(donors, recipients, match_args={"n_bins": 7})
    fused = pim.transform(donors, recipients)

    assert isinstance(pim.results["target"]["matches"], pd.DataFrame)
    tst.assert_equal(len(fused), len(recipients))
    assert "target" in fused.columns


def test_configurable_model_args():
    for method, m_args in [("linear", {"C": 100, "fit_intercept": False}),
                           ("tree", {"max_depth": 10, "max_leaf_nodes": 3}),
                           ("forest", {"n_estimators": 5, "criterion": "entropy"})]:
        pim = PIM("target", model_method=method)
        pim.fit(donors, recipients, model_args=m_args)
        assert isinstance(pim.results["target"]["matches"], pd.DataFrame)
