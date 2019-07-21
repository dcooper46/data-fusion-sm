# -*- coding: utf-8 -*-
"""
util functions related to information theory
"""

import numpy as np
from .util import get_counts, get_probs


def gini(x):
    """
    gini index to measure impurity of a distribution.
    maximal when values occur equally (equal probability).
    \n.. math::
      1 - \\sum_{i=0}^{n} x_i^2
    """
    _, counts = get_counts(x)
    p = get_probs(counts)

    return 1 - (p**2).sum()


def entropy(x, base=None):
    """
    measure of information that can be gained from the distribution.
    maximal when values occur equally (equal probability).
    \n.. math::
      H(X) = -\sum_{i=0}^{n} x_ilog_2{x_i}
    """
    _, counts = get_counts(x)
    p = get_probs(counts)
    if 1 in p:  # one class has 100% probability => all others 0.0
        return 0
    ret = -(p * np.log(p)).sum()
    if base is not None:
        ret /= np.log(base)
    return ret


def efficiency(x, base=None):
    """
    efficiency (i.e. normalized entropy) is a ratio that relates the
    entropy of a non-uniform distribution to a uniform one of same classes.
    \n.. math::
      E(X) = -\sum_{i=0}^{n} \\frac{x_ilog_2{x_i}}{log_2{n}}
    """
    _, counts = get_counts(x)
    p = get_probs(counts)

    ret = -(p * np.log(p) / np.log(len(p))).sum()
    if base is not None:
        ret /= np.log(base)
    return ret


def info_gain(x, y, impurity):
    """
    when impurity function is entropy, this is equivalent to Mutual Information
    \n.. math::
      IG(X, Y) = H(Y) - H(Y|X)
    """
    impurity_ygivenx = 0
    xvalues, xcounts = get_counts(x)
    xprobs = get_probs(xcounts)

    for xval, xprob in zip(xvalues, xprobs):
        impurity_ygivenx += xprob * impurity(y[x == xval])

    return impurity(y) - impurity_ygivenx


def cond_info_gain(x, y, z, impurity):
    """
    .. math::
      IG(X,Y|Z) = H(X|Z) + H(Y|Z) - H(X,Y|Z)
    """
    xy = list(zip(x, y))

    impurity_ygivenz = impurity_xgivenz = impurity_xygivenz = 0
    zvals, zcounts = get_counts(z)
    zprobs = get_probs(zcounts)

    for zval, zprob in zip(zvals, zprobs):
        xyz = [xyi[0] for xyi in zip(xy, z) if xyi[1] == zval]

        impurity_ygivenz += zprob * impurity(y[z == zval])
        impurity_xgivenz += zprob * impurity(x[z == zval])
        impurity_xygivenz += zprob * impurity(xyz)

    return impurity_xgivenz + impurity_ygivenz - impurity_xygivenz


def symmetrical_uncertainty(x, y):
    """
    \n.. math::
      SU = \\frac{2IG}{H(x) + H(y)}
    """
    return (2.0 * info_gain(x, y, entropy) / (entropy(x) + entropy(y)))


def info_gain_ratio(x, y, impurity):
    """
    impurity is callable function to calculate impurity (i.e. gini or entropy)
    \n.. math::
      IGR = \\frac{IG(X, Y)}{H(X)}
    """
    return info_gain(x, y, impurity) / impurity(x)


def _midd(x, y):
    return -entropy(list(zip(x, y))) + entropy(x) + entropy(y)


def _cmidd(x, y, z):
    return (entropy(list(zip(y, z))) + entropy(list(zip(x, z))) -
            entropy(list(zip(x, y, z))) - entropy(z))


def _xsi(x, n):
    """
    x is the count of observations
    similar to entropy
    """
    if x == 0:
        return 0
    else:
        return x / n * np.log(x)


_vxsi = np.vectorize(_xsi)


def _ent_hat(x):
    """ binary pseudo entropy """
    n = np.shape(x)[0]
    _, counts = get_counts(x)
    return np.log(n) - np.sum(vxsi(counts, n))


def _info_gain_hat(x, y):
    return _ent_hat(x) + _ent_hat(y) - _ent_hat(list(zip(x, y)))


def cond_info_gain_hat(x, y, z):
    return (_ent_hat(list(zip(y, z))) + _ent_hat(list(zip(x, z))) -
            _ent_hat(list(zip(x, y, z))) - _ent_hat(z))
