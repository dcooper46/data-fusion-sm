""" functions to compare discrete probability distributions """

import numpy as np
import pandas as pd


def kl_divergence(p, q):
    """
    `Kullback-Leibler Divergence <https://en.wikipedia.org/wiki/Kullback–Leibler_divergence>`_
    """
    return -sum(np.log(q / p) * p)


def hellinger(p, q):
    """
    `Hellinger distance <https://en.wikipedia.org/wiki/Hellinger_distance>`_
    """
    return np.sqrt(1 - sum(np.sqrt(p * q)))


def total_variation(p, q):
    """
    `Total Variation <https://en.wikipedia.org/wiki/Total_variation_distance_of_probability_measures>`_
    also called `statistical distance` or `variational distance`
    """
    return 0.5 * sum(np.abs(abs(p - q)))


def overlap(p, q):
    """
    `Overlap`
    The amount two distributions agree (1 - total_variation)
    """
    return 1 - total_variation(p, q)


def compare_distributions(p, q):
    """
    comparison summary between two distributions detailing
    how close they are.
    """
    comparisons = dict()
    comparisons["KL-Divergence"] = kl_divergence(p, q)
    comparisons["Hellinger Distance"] = hellinger(p, q)
    comparisons["Total Variation Distance"] = total_variation(p, q)
    comparisons["Overlap"] = overlap(p, q)
    return pd.Series(comparisons)
