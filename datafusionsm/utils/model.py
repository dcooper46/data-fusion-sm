"""
helper methods for building model wrappers
"""
from numpy import argmax
from pandas.api.types import is_numeric_dtype, is_string_dtype
from sklearn.base import clone
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


MODEL_MAP = {
    "linear": {
        "classification": LogisticRegression(solver='lbfgs', multi_class='multinomial'),
        "regression": LinearRegression()
    },
    "tree":  {
        "classification": DecisionTreeClassifier(),
        "regression": DecisionTreeRegressor()
    },
    "forest": {
        "classification": RandomForestClassifier(),
        "regression": RandomForestRegressor()
    }
}


def model_map(model_type, task_type):
    return MODEL_MAP[model_type][task_type]


def get_model(method, target, **model_args):
    task = define_task(target)
    return model_map(method, task).set_params(**model_args)


def train_model(model, X, y, k):
    if (model._estimator_type == "classification" and is_string_dtype(y)):
        le = LabelEncoder()
        y = le.fit_transform(y)
    ret_model = None
    if k > 1:
        kf = KFold(k, shuffle=True)
        models, scores = [], []
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            mod = clone(model).fit(X_train, y_train)
            models.append(mod)
            scores.append(mod.score(X_test, y_test))
        ret_model = models[argmax(scores)]
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        ret_model = clone(model).fit(X_train, y_train)
    return ret_model


def predict(model, X):
    if (model._estimator_type == "classifier" and
            hasattr(model, "predict_proba")):
        return model.predict_proba(X)
    else:
        return model.predict(X)


def define_task(target):
    if is_numeric_dtype(target):
        if target.nunique() / target.count() < 0.05:
            target_type = 'classification'
        else:
            target_type = 'regression'
    else:
        target_type = 'classification'
    return target_type
