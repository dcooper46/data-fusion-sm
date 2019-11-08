"""
Perform  Implicit Data Fusion by matching data records according
to a similarity score based on common linking variables.
"""
from __future__ import division, absolute_import

from collections import Counter

import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances

from fusion.implicit import neighbors as nb
from fusion.implicit import linear_assignment as la
from fusion.implicit import importance_weights as iw
from fusion.utils.exceptions import NotFittedError, DataFormError
from fusion.data.formatting import encode_columns


MATCH_METHODS = ["nearest", "neighbors", "hungarian", "jonker_volgenant"]

LOCAL_SCORE_METHODS = {"gower": None, "exact": None}

SKLEARN_SCORE_METHODS = {"cosine", "euclidean", "manhattan"}

IMPORTANCE_METHODS = ["entropy", "gini", "info_gain", "linear", "tree"]


class HotDeck:
    """
    fuse two data sources together using
    information implicit to each

    Parameters
    ----------
    match_method: str (default='nearest')
        'nearest', 'neighbors', 'hungarian', 'jonker_volgenant'
        algorithm used to match records from each data source

    score_method: str, callable, optional (default 'euclidean')
        similarity/distance measure to compare records.
        Can be any metric available in
        `scipy.spatial.distance
        <https://docs.scipy.org/doc/scipy/reference/spatial.distance.html#module-scipy.spatial.distance>`_
        or
        `sklearn.metrics <https://scikit-learn.org/stable/modules/classes.html#pairwise-metrics>`_

    minimize: boolean (default=True)
        minimize distance between records or maximize score/similarity

    importance: str (default=None)

    Attributes
    ----------
    critical: array-like[str]

    matches: array-like[tuple[str, str]]

    usage: Counter

    imp_wgts: array-like[float]
    """
    def __init__(
            self, match_method="nearest", score_method="jaccard",
            minimize=True, importance=None
    ):
        self.match_method = match_method
        if score_method in LOCAL_SCORE_METHODS:
            self.score_method = LOCAL_SCORE_METHODS[score_method]
        else:
            self.score_method = score_method
        self.importance = importance
        self.minimize = minimize
        self.linking = None
        self.critical = None
        self.matches = None
        self.scores = None
        self.usage = None
        self.imp_wgts = None

    def fit(self, donors, recipients,
            linking=None, critical=None, imp_wgts=None,
            target=None, match_args=None, score_args=None):
        """
        Fuse two data sources by matching records

        Parameters
        ----------
        donors: pandas.DataFrame
            Records containing information to be donated

        recipients: pandas.DataFrame
            Records that will receive information

        linking: array-like, optional (default=None)
            List of columns that will link the two data sources
            if None, all overlapping columns will be used

        critical: array-like, optional (default=None)
            Features that must match exactly when fusing

        imp_wgts: array-like, dict, optional(default=None)
            Importance weights for linking variables

        target: string, array-like, optional (default=None)
            What information will be donated. When given here, serves
            as the reference column if calculating importance weights.
            If None, importance weights will be by individual columns.

        match_args: dict, optional (default=None)
            Additional arguments for matching algorithm
            See the modules in :mod:`fusion.implicit.matching` for the
            list of possible matching parameters.

        score_args: dict, optional (default=None)
            Additional arguments for scoring method
            For a list of scoring functions that can be used,
            look at `sklearn.metrics`.

        Returns
        -------
        self: object

        Notes
        -----
        The data contained in donors and recipients is assumed to have
        at least a few overlapping features with common values.  They should
        also contain an `id` column appropriately titled.
        """
        kwargs = {"match_args": match_args if match_args else {},
                  "score_args": score_args if score_args else {}}

        donrs, recips, linking = _check_data(
            donors, recipients, self.score_method, linking
        )

        self.linking = linking

        if critical:
            self.critical = [l for l in linking
                             if l.rsplit("_", 1)[0] in critical]
        else:
            self.critical = []

        if imp_wgts:
            try:
                self.imp_wgts = pd.Series(imp_wgts)
                self.imp_wgts.index = linking
            except ValueError:
                raise DataFormError(
                    "passed number of weights, %d, does not match number"
                    "of liking variables, %d" % (len(imp_wgts), len(linking))
                )
        else:
            if not self.importance:
                self.imp_wgts = pd.Series(
                    np.ones(len(self.linking) - len(self.critical)),
                    index=[l for l in linking if l not in self.critical]
                )
            else:
                donrs_iw = donrs.drop(
                    self.critical, axis=1) if critical else donrs
                recips_iw = recips.drop(
                    self.critical, axis=1) if critical else recips
                if target:
                    imp_wgts = iw.s_iw(donrs_iw, target, self.importance)
                else:
                    imp_wgts = iw.u_iw(
                        pd.concat([donrs_iw, recips_iw]), self.importance
                    )
                self.imp_wgts = imp_wgts

        kwargs["score_args"]["method"] = self.score_method
        kwargs["score_args"]["wgts"] = self.imp_wgts
        kwargs["match_args"]["method"] = self.match_method
        kwargs["match_args"]["minimize"] = self.minimize

        matches, scores, usage = _get_matches(donrs, recips, self.critical, **kwargs)
        self.matches = pd.DataFrame(matches, columns=["recipient_id", "donor_id"])
        self.scores = scores
        self.usage = usage
        return self

    def transform(self, donors, recipients, target=None):
        """
        Using fused ids, impute information from donor data to the
        recipient data.

        Parameters
        ----------
        donors: pandas.DataFrame
            Records containing information to be donated

        recipients: pandas.DataFrame
            Records that will receive information

        target: string, array-like, optional (default=None)
            What information will be shared from donor data.
            If None, data will be joined on fused indices and all
            overlapping fields from `donors` will have `_d` suffix.

        Returns
        -------
        ret: pandas.DataFrame
            New DataFrame containing dontated information
        """
        if self.matches is None:
            raise NotFittedError(
                "This PpcFusion instance is not fitted yet. Call 'fit' with"
                "appropriate arguments before using this method."
            )
        recip_ids, donor_ids = zip(*self.matches)
        matched_donors = donors.iloc[list(donor_ids)].reset_index(drop=True)
        out_recips = recipients.iloc[list(recip_ids)].reset_index(drop=True)
        if target:
            ret = out_recips.join(matched_donors[target])
        else:
            ret = out_recips.join(matched_donors, rsuffix="_d")
        return ret

    def fit_transform(self, donors, recipients, linking=None, critical=None,
                      target=None, match_args=None, score_args=None):
        """
        Fit fusion model and transform recipient data with donated info
        Short-cut for cls.fit().transform()
        """
        self.fit(
            donors, recipients, linking, critical,
            target, match_args, score_args
        )
        return self.transform(donors, recipients, target)


def _check_data(donors, recipients, score_method, linking=None):
    """
    ensure data to be fused is appropriate for requested methods
    """
    if not linking:
        d_cols = donors.columns
        r_cols = recipients.columns
        linking = list(set(d_cols).intersection(r_cols))
    if score_method not in ["gower", "exact"]:
        donors, d_ecols = encode_columns(donors[linking])
        recipients, r_ecols = encode_columns(recipients[linking])
        linking = list(set(d_ecols).intersection(r_ecols))
    return donors[linking], recipients[linking], linking


def _get_matches(donors, recipients, critical, **kwargs):
    if critical is None or critical == []:
        donors["groupby"] = -1
        recipients["groupby"] = -1
        critical = "groupby"
    matches, scores = _match_critical(donors, recipients, critical, **kwargs)
    _, matched_donors = zip(*matches)
    usage = Counter(matched_donors)
    return matches, scores, usage


def _match_critical(donors, recipients, critical, **kwargs):
    donor_cells = donors.groupby(critical, sort=False)
    recip_cells = recipients.groupby(critical, sort=False)
    matches = []
    scores = []
    for cc_idx, r_cc in recip_cells:
        d_cc = donor_cells.get_group(cc_idx).drop(critical, axis=1)
        r_cc = r_cc.drop(critical, axis=1)
        d_cc_ids = donor_cells.indices[cc_idx]
        r_cc_ids = recip_cells.indices[cc_idx]
        cell_matches, cell_scores = _match_full(d_cc, r_cc, **kwargs)
        cell_matched_ids = [(r_cc_ids[i], d_cc_ids[j])
                            for i, j in cell_matches]
        matches.extend(cell_matched_ids)
        scores.extend(cell_scores)
    return matches, scores


def _match_full(donors, recipients, **kwargs):
    score_args = kwargs["score_args"]
    score_method = score_args["method"]
    imp_wgts = score_args["wgts"]
    _score_args = {k: v for k, v in score_args.items()
                   if k not in ("method", "wgts")}
    scores_mat = _score_records(
        recipients, donors, score_method, imp_wgts, **_score_args
    )

    match_args = kwargs["match_args"]
    match_method = match_args["method"]
    minimize = match_args["minimize"]
    _match_args = {k: v for k, v in match_args.items()
                   if k not in ("method", "minimize")}
    if match_method not in MATCH_METHODS:
        raise ValueError(
            "invalid matching method."
            "Possible values are {}".format(",".join(MATCH_METHODS))
        )
    if match_method == 'nearest':
        matches = nb.nearest(scores_mat, minimize, **_match_args)
    elif match_method == 'neighbors':
        try:
            k = _match_args.pop('k')
            matches = nb.neighbors(
                scores_mat, k, minimize, **_match_args
            )
        except KeyError:
            raise ValueError(
                "neighborhood matching selected but"
                "missing value for the number of neighbors"
            )
    elif match_method in la.methods:
        matches, scores = la.lap(match_method, scores_mat, minimize)
    else:
        raise NotImplementedError(
            "the chosen method is not currently"
            "available: {}".format(match_method)
        )
    return matches, scores


def _score_records(x, y, metric, wgts, **kwargs):
    if metric in SKLEARN_SCORE_METHODS:
        if metric == 'manhattan':
            wgt_scl = wgts
        else:
            wgt_scl = np.sqrt(wgts)
        return pairwise_distances(
            x*wgt_scl, y*wgt_scl, metric,
            n_jobs=kwargs.get("n_jobs", -1), **kwargs
        )
    return pairwise_distances(
        x, y, metric,
        n_jobs=kwargs.get("n_jobs", -1),
        w=wgts, **kwargs
    )
