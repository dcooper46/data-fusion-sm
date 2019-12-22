"""
various methods for calculating importance
weights between variables, and possibly a target
"""

from functools import partial

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_string_dtype
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

from fusion.utils import information as it


def supervised_imp_wgts(data, target, method, **kwargs):
    """
    Create feature importance weights with respect to a target variable

    Parameters
    ----------
    data: pandas.DataFrame
        input data with features to create weights for

    target: str, array-like
        reference target in determining weights

    method: str
        weighting method: 'info_gain', 'mutual_information', 'linear', 'tree'

    Returns
    -------
    wgts: array-like
        feature weights in same order as input
    """
    _target = data[target] if isinstance(target, str) else target
    _data = data.drop(target, axis=1) if isinstance(target, str) else data
    if method in ('info_gain', 'mutual_information'):
        if 'impurity_func' in kwargs:
            impurity = getattr(it, kwargs.get('impurity_func'))
        else:
            impurity = it.entropy
        wgts = []
        for c in _data.columns:
            wgts.append(it.info_gain(_data[c], _target, impurity))
        wgts = pd.Series(wgts, index=_data.columns)
    elif method in ('linear', 'tree'):
        model = _get_model(method, _data, _target)
        wgts = model.feature_importances
    else:
        wgts = pd.Series(np.ones(len(_data.columns)), index=_data.columns)
    return wgts


def unsupervised_imp_wgts(data, method, **kwargs):
    """
    Create feature importance weights base on each individual feature

    Parameters
    ----------
    data: pandas.DataFrame
        input data with features to create weights for

    method: str
        weighting method: 'info_gain', 'mutual_information', 'linear', 'tree'

    Returns
    -------
    wgts: array-like
        feature weights in same order as input
    """
    # TODO: handle different data types (numeric or categorical)
    if method in ('entropy', 'efficiency'):
        func = partial(it.entropy, base=kwargs.get("ebase"))
        wgts = data.apply(func, axis=0)
    else:
        wgts = data.apply(it.gini, axis=0)
    if method in ('entropy', 'gini', 'efficiency'):
        # maximal entropy/gini is not as informative (uniform distribution)
        wgts = _scale_wgts(wgts)
    return wgts


def _scale_wgts(wgts):
    names = wgts.index
    zmask = wgts == 0
    scaled_wgts = np.empty_like(wgts)
    scaled_wgts[zmask] = 0
    scaled_wgts[~zmask] = 1 / wgts[~zmask]
    return pd.Series(scaled_wgts, index=names)


def _get_model(method, data, target):
    task = _define_task(target)
    if task == 'classification':
        if is_string_dtype(target):
            le = LabelEncoder()
            target = le.fit_transform(target)
    model = FeatureModel(method, task)
    model.fit(data, target)
    return model


def _define_task(target):
    if is_numeric_dtype(target):
        if target.nunique() / target.count() < 0.05:
            target_type = 'classification'
        else:
            target_type = 'regression'
    else:
        target_type = 'classification'
    return target_type


class FeatureModel:
    """
    simple model framework to train a predictive model
    and store feature weights

    Parameters
    ----------
    model_type: str
        'linear' or 'tree'

    task_type: str
        'classification' or 'regression'

    Attributes
    ----------
    model: object
        fitted model used for creating feature weights

    features: array-like
        features used for fitting model
    """
    def __init__(self, model_type, task_type):
        self.model_type = model_type
        self.task_type = task_type
        self.model = MODEL_MAP[model_type][task_type]
        self.features = None

    def fit(self, data, target):
        self.model.fit(data, target)
        self.features = data.columns
        return self

    @property
    def feature_importances(self):
        if hasattr(self.model, "feature_importances_"):
            ret = pd.Series(self.model.feature_importances_, index=self.features)
        else:
            imps = np.mean(np.abs(self.model.coef_), axis=0)
            ret = pd.Series(imps, index=self.features)
        return ret


MODEL_MAP = {
    "linear": {
        "classification": LogisticRegression(solver='lbfgs', multi_class='multinomial'),
        "regression": LinearRegression()
    },
    "tree":  {
        "classification": DecisionTreeClassifier(),
        "regression": DecisionTreeRegressor()
    }
}
