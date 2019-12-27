"""
methods to perform statistical
matching (hot-deck-matching) between
datasets based on record similarity/distances
"""
import operator
import numpy as np

np.seterr(all='print')


def _scale_penalty(x, flatness):
    return np.sinh(x) / (flatness * np.cosh(x))


def neighbors(scores_mat, k=5, minimize=True, **kwargs):
    """
    select matching pair from neighborhood of possible matches

    Parameters
    ----------
    scores: array-like, shape: (n_recipients, n_donors)
        matrix of similarity scores

    k: int, float

    minimize: boolean

    Returns
    -------
    matches: list
        final matching pairs of records
    """
    nrecips, ndonors = scores_mat.shape
    matches = []
    matched_scores = []
    used = np.ones(ndonors)

    comp = operator.le if minimize else operator.ge
    score_update = operator.mul if minimize else operator.truediv
    sort_order = 1 if minimize else -1

    for recip_id in range(nrecips):
        scores = score_update(scores_mat[recip_id], used)
        if minimize and 0 in scores:  # neither * nor / will affect scores
            scores += _scale_penalty(used - 1, kwargs.get('flatness', 3))
        if isinstance(k, float):
            p = k if k >= 1 else k * 100
            neighborhood = np.where(comp(scores, np.percentile(scores, p)))[0]
        elif isinstance(k, int):
            neighborhood = np.argsort(sort_order * scores)[:k]
        else:
            raise ValueError("unrecognized value for k: {}".format(k))
        matched_id = np.random.choice(neighborhood, p=kwargs.get('probs'))
        matches.append((recip_id, matched_id))
        matched_scores.append(scores[matched_id])
        used[matched_id] += kwargs.get("penalty", 1)

    return matches, matched_scores


def nearest(scores_mat, minimize=True, **kwargs):
    """
    select highest scoring ids as matches

    Parameters
    ----------
    scores: array-like, shape: (n_recipients, n_donors)
        matrix of similarity scores

    minimize: boolean

    Returns
    -------
    matches: list
        final matching pairs of records
    """
    return neighbors(scores_mat, 1, minimize, **kwargs)
