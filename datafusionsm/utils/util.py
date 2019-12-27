# -*- coding: utf-8 -*-
""" utility functions used in other modules """

from __future__ import division, print_function

from collections import Counter

import numpy as np
import pandas as pd


def check_array(array):
    """
    check for DataFrame and convert object types to ints where necessary
        - if given numpy array, assume uniform numeric types for now
    """
    array_type = type(array).__name__
    if array_type == 'DataFrame':
        cat_cols = array.select_dtypes(exclude=[np.number]).columns
        array[cat_cols] = array[cat_cols].apply(lambda x: x.astype('category'))
        array[cat_cols] = array[cat_cols].apply(lambda x: x.cat.codes)
        return array.as_matrix()
    elif array_type == 'Series':
        if array.dtype.kind == 'O':
            array = pd.Categorical(array)
            array = array.codes
        return array
    else:
        # for numpy array, assume uniform data types for now
        return array


def check_X_y(X, y, multi_output=False):
    X = check_array(X)
    if multi_output:
        y = check_array(y)
    else:
        yshape = y.shape
        if len(yshape) == 1:
            y = np.ravel(y)
        elif len(yshape) == 2 and yshape[1] == 1:
            y = np.ravel(y)
        else:
            raise ValueError("y has bad shape {}. Expecting single "
                             "vector".format(yshape))
    return X, y


def check_association_inputs(data: pd.DataFrame, target):
    if isinstance(target, str):
        ret_target = data[target]
        ret_data = data.drop(target, axis=1)
    elif isinstance(target, int):
        ret_target = data[data.columns[target]]
        ret_data = data.drop(data.columns[target], axis=1)
    elif isinstance(target, (pd.Series, np.ndarray)):
        ret_target = target
    else:
        raise ValueError(
            "invalid target given, {} with dataframe".format(target))
    return ret_data, ret_target


def get_counts(x):
    """
    count each distinct value in a discrete array/series.

    Parameters
    ----------
    x: array-like

    Returns
    -------
    array-like, array-like
        counts of each value in the series
    """
    # return np.unique(x, return_counts=True)
    values, counts = zip(*Counter(x).items())
    return np.array(values), np.array(counts)


def get_probs(x):
    """
    calculate sample probability of each distinct value
    in a discrete array/series.

    Parameters
    ----------
    x: pandas series or numpy array containing value counts

    Returns
    -------
    sample probability of each value
    """
    return x / x.sum()


def ensure_percentage(p):
    """
    makes sure percentage p is in range 0-1
    """
    if 0 <= p <= 1:
        return float(p)
    elif 0 <= p <= 100:
        return float(p) / 100.0
    else:
        raise ValueError("{} is not a valid percentage".format(p))
