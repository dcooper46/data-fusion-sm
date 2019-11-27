from numpy import testing as tst
import pandas as pd
import pytest
from sklearn.svm import SVC

from fusion.implicit_model import PMM
from fusion.utils.formatting import encode_columns


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


def test_default_setup():
    pmm = PMM("target")

    tst.assert_equal(pmm.match_method, "nearest")
    tst.assert_equal(pmm.score_method, "euclidean")
    tst.assert_equal(pmm.model_method, "linear")

    pmm.fit(donors, recipients)

    tst.assert_array_equal(pmm.critical, [])
    tst.assert_equal(len(pmm.linking), 6)
    tst.assert_equal(list(pmm.results)[0], "target")
    tst.assert_equal(len(recipients), len(pmm.results["target"]["matches"]))

    fused = pmm.transform(donors, recipients)
    tst.assert_equal(len(fused), len(recipients))
    assert "target" in fused.columns


def test_default_fit_transform():
    pmm = PMM("target")
    fused = pmm.fit_transform(donors, recipients)
    tst.assert_equal(len(fused), len(recipients))
    tst.assert_equal(fused.shape[1], len(recipients.columns) + 1)


def test_critical_cells():
    pmm = PMM("target")
    pmm.fit(donors, recipients, critical="a")
    tst.assert_array_equal(sorted(pmm.critical), ["a_0", "a_1"])
    fused = pmm.transform(donors, recipients)

    assert isinstance(pmm.results["target"]["matches"], pd.DataFrame)
    tst.assert_equal(len(fused), len(recipients))
    assert "target" in fused.columns


def test_critical_cells_with_1_kfold():
    pmm = PMM("target")
    model_args = {"k": 1}
    pmm.fit(donors, recipients, critical="a", model_args=model_args)
    tst.assert_array_equal(sorted(pmm.critical), ["a_0", "a_1"])
    fused = pmm.transform(donors, recipients)

    assert isinstance(pmm.results["target"]["matches"], pd.DataFrame)
    tst.assert_equal(len(fused), len(recipients))
    assert "target" in fused.columns


def test_custom_model_probability_output():
    svc = SVC(probability=True)
    svc.fit(encoded_donors[linking], donors["target"])

    pmm = PMM("target")
    model_args = {"model": svc}
    pmm.fit(donors, recipients, model_args=model_args)
    fused = pmm.transform(donors, recipients)

    # model trained within critical cells
    svc.fit(encoded_donors[linking].drop(["a_1", "a_0"], axis=1), donors["target"])
    pmm.fit(donors, recipients, critical="a", model_args=model_args)
    fused = pmm.transform(donors, recipients)

    assert isinstance(pmm.results["target"]["matches"], pd.DataFrame)
    tst.assert_equal(len(fused), len(recipients))
    assert "target" in fused.columns


def test_custom_model_regression_output():
    svc = SVC()
    svc.fit(encoded_donors[linking], donors["target"])

    pmm = PMM("target")
    model_args = {"model": svc}
    pmm.fit(donors, recipients, model_args=model_args)
    fused = pmm.transform(donors, recipients)

    # model trained within critical cells
    svc.fit(encoded_donors[linking].drop(
        ["a_1", "a_0"], axis=1), donors["target"])
    pmm.fit(donors, recipients, critical="a", model_args=model_args)
    fused = pmm.transform(donors, recipients)

    assert isinstance(pmm.results["target"]["matches"], pd.DataFrame)
    tst.assert_equal(len(fused), len(recipients))
    assert "target" in fused.columns


def test_configurable_scoring_methods():
    for scorer in ["cosine", "euclidean", "manhattan"]:
        pmm = PMM("target", score_method=scorer)
        pmm.fit(donors, recipients)
        assert isinstance(pmm.results["target"]["matches"], pd.DataFrame)


def test_configurable_matching_methods():
    for matcher in ["nearest", "neighbors", "hungarian", "jonker_volgenant"]:
        pmm = PMM("target", match_method=matcher)
        pmm.fit(donors, recipients)
        assert isinstance(pmm.results["target"]["matches"], pd.DataFrame)


def test_configurable_scoring_args():
    pmm = PMM("target", score_method="euclidean")
    pmm.fit(donors, recipients, score_args={"squared": True})
    assert isinstance(pmm.results["target"]["matches"], pd.DataFrame)


def test_configurable_matching_args():
    pmm = PMM("target", match_method="neighbors")
    pmm.fit(donors, recipients, match_args={"k": 7, "flatness": 5})
    assert isinstance(pmm.results["target"]["matches"], pd.DataFrame)


def test_configurable_model_args():
    for method, m_args in [("linear", {"C": 0.5, "l1_ratio": 0}),
                           ("tree", {"max_depth": 10, "max_leaf_nodes": 3}),
                           ("forest", {"n_estimators": 5, "criterion": "entropy"})]:
        pmm = PMM("target", model_method=method)
        pmm.fit(donors, recipients, model_args=m_args)
        assert isinstance(pmm.results["target"]["matches"], pd.DataFrame)
