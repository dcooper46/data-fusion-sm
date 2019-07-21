"""
Methods that solve matching as a
`linear assignment problem <https://en.wikipedia.org/wiki/Assignment_problem>`_
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
# from lapjv import lapjv


def hungarian(scores_mat, minimize=True):
    """
    find optimal assignments between donors and recipients
    by max/minimizing overall score (not individually) by
    using the Hungarian algorithm.

    This method ensures all recipients are given a donor,
    which means donors can be used multiple times.

    Parameters
    ----------
    scores_mat: array-like, shape (nrecipients, ndonors)
        matrix of scores/distances between records

    minimize: boolean, optional (default=True)
        minimize overall score?

    Returns
    -------
    matches: list[[int, int]]
        [row, col] indices of assignments
    """
    if not minimize:
        scores_mat = scores_mat.max() - scores_mat
    nrecip, _ = scores_mat.shape
    matches = []
    available = np.arange(nrecip)
    _scores_mat = scores_mat
    while _scores_mat.shape[0] > 0:
        _scores_mat = _scores_mat[available]
        recip_mindexs, donor_mindexs = linear_sum_assignment(_scores_mat)
        matches.extend(
            np.column_stack((available[recip_mindexs], donor_mindexs))
        )
        available = np.delete(available, recip_mindexs)
    return matches


def jonker_volgenant(scores_mat, minimize=True):
    """
    find optimal assignments between donors and recipients
    by max/minimizing overall score (not individually) using
    the Jonker-Volgenant algorithm (a faster alternative to
    the Hungarian algorithm).

    This implementation ensures all recipients are given
    a donor, which means donors can be used multiple times.

    Parameters
    ----------
    scores_mat: array-like, shape (nrecipients, ndonors)
        matrix of scores/distances between records

    minimize: boolean, optional (default=True)
        minimize overall score?

    Returns
    -------
    matches: list[[int, int]]
        [row, col] indices of assignments
    """
    # implementation requires a square cost matrix
    nrecip, ndonor = scores_mat.shape
    if nrecip != ndonor:
        donor_ids = np.arange(ndonor)
        avg_donor_scores = np.mean(scores_mat, axis=1)
        sort_order = 1 if minimize else -1
        top_donors = np.argsort(sort_order * avg_donor_scores)
        if nrecip < ndonor:
            # remove donors
            select_donors = top_donors[:nrecip]
            row_ind, col_ind, _ = lapjv(scores_mat[:, select_donors])
            matches = np.column_stack((row_ind, donor_ids[col_ind]))
        elif nrecip > ndonor:
            # add donors
            n_add_ons = nrecip - ndonor
            select_donors = np.concatenate((donor_ids, top_donors[:n_add_ons]))
            row_ind, col_ind, _ = lapjv(scores_mat[:, select_donors])
            matches = np.column_stack((row_ind, donor_ids[col_ind]))
    else:
        row_ind, col_ind, _ = lapjv(scores_mat)
        matches = np.column_stack((row_ind, col_ind))
    return matches
