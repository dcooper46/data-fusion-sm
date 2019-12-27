"""
methods to compare categorical variables with statistical measures
"""
import numpy as np
import pandas as pd

from datafusionsm.utils.information import contingency_info_gain
from datafusionsm.utils.util import check_association_inputs


def pairwise_associations(data, target):
    """
    compare independent variables with a target variable to
    determine which have the strongest associations, and thus
    will likely be good predictors and linking variables.

    Parameters
    ----------
    data: pandas.DataFrame
        dataset with independent variables to compare

    target: str, int, pandas.Series
        target to compare against

    Returns
    -------
    ret: pandas.DataFrame
        summary of comparisons with the target with covariates as rows
    """
    data, target = check_association_inputs(data, target)
    associations = dict()
    for predictor in data.columns:
        crosstabs = np.asanyarray(pd.crosstab(data[predictor], target))
        cv = cramers_v(crosstabs)
        gkl = goodman_kruskal_lambda(crosstabs)
        gkt = goodman_kruskal_tau(crosstabs)
        tu = theils_u(crosstabs)
        associations[predictor] = {
            "CramersV": cv,
            "GoodmanKruskal-Lambda": gkl,
            "GoodmanKruskal-Tau": gkt,
            "Theil's Uncertainty": tu
        }
    return pd.DataFrame.from_dict(associations, orient='index')


def chi_sqr(observed):
    """
    Chi Square test statistic given observed data
    assumes 2 dims for now
    """
    marginals = [observed.sum(axis=a) for a in range(observed.ndim)]
    expected = np.outer(*marginals) / observed.sum()
    return np.sum((observed.T - expected)**2 / expected)


# Cramers V
def cramers_v(xtabs):
    """
    see:
    `Cramer's V <https://en.wikipedia.org/wiki/Cramér%27s_V>`_
    """
    xtabs = np.asanyarray(xtabs)
    n = xtabs.sum()
    chi2stat = chi_sqr(xtabs)
    phi2 = chi2stat / n
    r, k = xtabs.shape
    phi2t = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    kt = k - ((k-1)**2 / (n-1))
    rt = r - ((r-1)**2 / (n-1))
    v = np.sqrt(phi2t / min(kt-1, rt-1))
    return v


def goodman_kruskal_lambda(xtabs):
    """
    see:
    `Goodman Kruskal Lambda <https://en.wikipedia.org/wiki/Goodman_and_Kruskal%27s_lambda>`_
    `Lambda example <http://www.vassarstats.net/lamexp.html>`_
    """
    return _goodman_kruskal(xtabs, _calc_gk_lambda)


def goodman_kruskal_tau(xtabs):
    """
    see:
    `Goodman Kruskal Tau <https://cran.r-project.org/web/packages/GoodmanKruskal/vignettes/GoodmanKruskal.html>`_
    """
    return _goodman_kruskal(xtabs, _calc_gk_tau)


def _goodman_kruskal(xtabs, stat_func):
    # row | col (x | y)
    return stat_func(xtabs, 0)


def _calc_gk_lambda(xtabs, base_axis):
    n = xtabs.sum()
    informed_axis = 0 if base_axis else 1
    base_error_prob = 1 - xtabs.sum(axis=base_axis).max() / n
    informed_error_prob = 1 - xtabs.max(axis=informed_axis).sum() / n
    error_reduction = (base_error_prob - informed_error_prob) / base_error_prob
    return error_reduction


def _calc_gk_tau(xtabs, base_axis):
    ptabs = xtabs / xtabs.sum()
    informed_axis = 0 if base_axis else 1
    marginals = [ptabs.sum(axis=a) for a in range(ptabs.ndim)]

    var_base = 1 - sum(marginals[base_axis]**2)
    ptabs_base_sqr = np.sum(ptabs**2, axis=informed_axis)
    expected_var_base = 1 - sum(ptabs_base_sqr / marginals[informed_axis])

    return (var_base - expected_var_base) / var_base


def theils_u(xtabs):
    """
    `Theils Uncertainty <https://en.wikipedia.org/wiki/Uncertainty_coefficient>`_
    \n.. math::

    U(X|Y)={\\frac  {H(X)-H(X|Y)}{H(X)}}={\\frac  {I(X;Y)}{H(X)}},
    """
    return contingency_info_gain(xtabs, norm=True)[0]
