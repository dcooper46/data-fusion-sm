"""
Functions to compare discrete probability distributions.  All methods take
two array-like probability distributions as input.
"""

import numpy as np
import pandas as pd


def kl_divergence(p, q):
    """
    `Kullback-Leibler Divergence <https://en.wikipedia.org/wiki/Kullback–Leibler_divergence>`_:
    :math:`KL(P || Q) = -\sum_{i=0}^{n}\log{\\frac{q_i}{p_i} * p_i}`
    """
    return -sum(np.log(q / p) * p)


def hellinger(p, q):
    """
    `Hellinger distance <https://en.wikipedia.org/wiki/Hellinger_distance>`_:
    :math:`H(P,Q) = \sqrt{1 - \\sum_{i=0}^{n}\sqrt{p_i * q_i}}`
    """
    return np.sqrt(1 - sum(np.sqrt(p * q)))


def total_variation(p, q):
    """
    `Total Variation <https://en.wikipedia.org/wiki/Total_variation_distance_of_probability_measures>`_:
    :math:`\delta(P,Q) = {\\frac {1}{2}}\|P-Q\|_{1} = {\\frac {1}{2}}\sum _{\omega \in \Omega }|P(\omega )-Q(\omega )|`

    Also called `statistical distance` or `variational distance`
    """
    return 0.5 * sum(np.abs(abs(p - q)))


def overlap(p, q):
    """
    `Overlap`
    The amount two distributions agree (1 - total_variation)
    """
    return 1 - total_variation(p, q)


def compare_dists(p, q):
    """
    A comparison summary between two distributions detailing how close
    they are. Returned as a pandas Series with the measures as index names.
    """
    comparisons = dict()
    comparisons["KL-Divergence"] = kl_divergence(p, q)
    comparisons["Hellinger Distance"] = hellinger(p, q)
    comparisons["Total Variation Distance"] = total_variation(p, q)
    comparisons["Overlap"] = overlap(p, q)
    return pd.Series(comparisons)
