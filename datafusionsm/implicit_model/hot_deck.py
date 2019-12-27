"""
data fusion via hot-deck imputation by
finding similar records and donating the live (thus hot)
information from one source to the other
"""
from __future__ import division, absolute_import

import warnings

import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances

from datafusionsm.implicit_model.base import BaseImplicitModel, ImplicitModelMixin
from datafusionsm.implicit_model import neighbors as nb
from datafusionsm.implicit_model import linear_assignment as la
from datafusionsm.importance_weights import (
    supervised_imp_wgts as s_iw, unsupervised_imp_wgts as u_iw
)
from datafusionsm.utils.exceptions import NotFittedError, DataFormError


class HotDeck(ImplicitModelMixin, BaseImplicitModel):
    """
    fuse two data sources together using
    statistical matching (hot-deck imputation) based on record
    similarity/distances and information implicit to each

    Parameters
    ----------
    match_method: str (default='nearest')
        'nearest', 'neighbors', 'hungarian', 'jonker_volgenant'
        algorithm used to match records from each data source

    score_method: str, callable, optional (default 'cosine')
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
            self, match_method="nearest", score_method="cosine",
            minimize=True, importance=None
    ):
        self.match_method = match_method
        if score_method in self.local_score_methods:
            self.score_method = self.local_score_methods[score_method]
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
            target=None, match_args=None, score_args=None,
            donor_id_col: int = 0, recipient_id_col: int = 0):
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

        ppc_id_col: int = 0
            Index of column serving as donor record index

        panel_id_col: int = 0
            Index of column serving as recipient record index

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

        donor_id = donors.columns[donor_id_col]
        recipient_id = recipients.columns[recipient_id_col]

        donrs, recips, linking = self.check_data(
            donors.drop(donor_id, axis=1),
            recipients.drop(recipient_id, axis=1),
            self.score_method,
            linking
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
            except ValueError as e:
                raise DataFormError(
                    "passed number of weights, %d, does not match number"
                    "of liking variables, %d: \n%s" % (
                        len(imp_wgts), len(linking), e
                    )
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
                    imp_wgts = s_iw(donrs_iw, target, self.importance)
                else:
                    imp_wgts = u_iw(
                        pd.concat([donrs_iw, recips_iw]), self.importance
                    )
                self.imp_wgts = imp_wgts

        kwargs["score_args"]["method"] = kwargs["score_args"].get("method", self.score_method)
        kwargs["score_args"]["wgts"] = self.imp_wgts
        kwargs["match_args"]["method"] = kwargs["match_args"].get("method", self.match_method)
        kwargs["match_args"]["minimize"] = self.minimize

        _matches, scores, usage = self.get_matches(donrs, recips, self.critical, **kwargs)
        recip_ids, donor_ids = zip(*_matches)
        matched_donors = donors.iloc[list(donor_ids)].reset_index(drop=True)[donor_id]
        matched_recipients = recipients.iloc[list(recip_ids)].reset_index(drop=True)[recipient_id]

        self._matches = pd.DataFrame(_matches, columns=[recipient_id, donor_id])
        self.matches = pd.DataFrame(zip(matched_recipients, matched_donors),
                                    columns=[recipient_id, donor_id])
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
        recip_ids, donor_ids = zip(*self._matches.values)
        matched_donors = donors.iloc[list(donor_ids)].reset_index(drop=True)
        out_recips = recipients.iloc[list(recip_ids)].reset_index(drop=True)
        if target:
            ret = out_recips.join(matched_donors[target])
        else:
            ret = out_recips.join(matched_donors, rsuffix="_d")
        return ret

    def _match_full(self, donors, recipients, **kwargs):
        score_args = kwargs["score_args"]
        score_method = score_args["method"]
        _score_args = {k: v for k, v in score_args.items()
                       if k != "method"}
        scores_mat = self._score_records(
            recipients, donors, score_method, **_score_args
        )

        match_args = kwargs["match_args"]
        match_method = match_args["method"]
        minimize = match_args["minimize"]
        _match_args = {k: v for k, v in match_args.items()
                       if k not in ("method", "minimize")}
        if match_method not in self.match_methods:
            raise ValueError(
                "invalid matching method."
                "Possible values are {}".format(",".join(self.match_methods))
            )
        if match_method == 'nearest':
            matches, scores = nb.nearest(scores_mat, minimize, **_match_args)
        elif match_method == 'neighbors':
            try:
                k = _match_args.pop('k')
                matches, scores = nb.neighbors(
                    scores_mat, k, minimize, **_match_args
                )
            except KeyError as e:
                warnings.warn(
                    "neighborhood matching selected but"
                    "missing value for the number of neighbors: " + str(e) +
                    "  Using default value of 5"
                )
                matches, scores = nb.neighbors(
                    scores_mat, minimize=minimize, **_match_args
                )
        elif match_method in la.methods:
            matches, scores = la.lap(match_method, scores_mat, minimize)
        else:
            raise NotImplementedError(
                "the chosen method is not currently"
                "available: {}".format(match_method)
            )
        return matches, scores

    def _score_records(self, x, y, metric, **kwargs):
        # possibly utilize spark here
        wgts = kwargs.pop("wgts")
        if metric in self.sklearn_score_methods:
            if metric == 'manhattan':
                wgt_scl = wgts
            else:
                wgt_scl = np.sqrt(wgts)
            return pairwise_distances(
                x*wgt_scl, y*wgt_scl, metric,
                n_jobs=kwargs.get("n_jobs", -1), **kwargs
            )
        return pairwise_distances(
            x.values, y.values, metric,
            n_jobs=kwargs.get("n_jobs", -1),
            w=wgts, **kwargs
        )
