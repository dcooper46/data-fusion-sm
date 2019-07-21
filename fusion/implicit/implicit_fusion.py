"""
Perform  Implicit Data Fusion by matching data records according
to a similarity score based on common linking variables.
"""
from __future__ import division, absolute_import

from collections import Counter

import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances

from fusion.implicit.matching import statistical_matching as sm
from fusion.implicit.matching import linear_assignment as la
from fusion.implicit.importance import importance_weights as iw
from fusion.util.exceptions import NotFittedError
from fusion.data.formatting import encode_columns


MATCH_METHODS = ["nearest", "neighbors", "hungarian", "jonker_volgenant"]

LOCAL_SCORE_METHODS = {"gower": None, "exact": None}

SKLEARN_SCORE_METHODS = {"cosine", "euclidean", "manhattan"}

IMPORTANCE_METHODS = ["entropy", "gini", "info_gain", "linear", "tree"]


class ImplicitFusion:
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
        self.critical = None
        self.matches = None
        self.usage = None
        self.imp_wgts = None

    def fit(self, donors, recipients,
            linking=None, critical=None, target=None,
            match_args=None, score_args=None):
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

        self.critical = critical

        donrs, recips = self._check_data(donors, recipients, linking)

        if not self.importance:
            self.imp_wgts = pd.Series(
                np.ones(len(self.linking) - len(self.critical)),
                index=[l for l in self.linking if l not in self.critical]
            )
        else:
            self.imp_wgts = self._calc_importance_weights(
                donrs, recips, target
            )

        self._get_matches(donrs, recips, **kwargs)
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

    def _check_data(self, donors, recipients, linking=None):
        """
        ensure data to be fused is appropriate for requested methods
        """
        if not linking:
            d_cols = donors.columns
            r_cols = recipients.columns
            linking = list(set(d_cols).intersection(r_cols))
        if self.score_method not in ["gower", "exact"]:
            donors, d_ecols = encode_columns(donors[linking])
            recipients, r_ecols = encode_columns(recipients[linking])
            linking = list(set(d_ecols).intersection(r_ecols))
            if self.critical:
                self.critical = [l for l in linking
                                 if l.rsplit("_", 1)[0] in self.critical]
        self.linking = linking
        return donors[linking], recipients[linking]

    def _calc_importance_weights(self, donors, recipients, target):
        if self.critical:
            donors = donors.drop(self.critical, axis=1)
            recipients = recipients.drop(self.critical, axis=1)
        if not target:
            ret = iw.unsupervised_imp_wgts(
                pd.concat([donors, recipients]), self.importance
            )
        else:
            ret = iw.supervised_imp_wgts(donors, target, self.importance)
        return ret

    def _get_matches(self, donors, recipients, **kwargs):
        critical = self.critical
        if critical is None or critical == []:
            donors["groupby"] = -1
            recipients["groupby"] = -1
            self.critical = "groupby"
        self.matches = self._match_critical(donors, recipients, **kwargs)
        _, matched_donors = zip(*self.matches)
        self.usage = Counter(matched_donors)
        self.critical = None if self.critical == "groupby" else self.critical

    def _match_critical(self, donors, recipients, **kwargs):
        donor_cells = donors.groupby(self.critical, sort=False)
        recip_cells = recipients.groupby(self.critical, sort=False)
        matches = []
        for cc_idx, r_cc in recip_cells:
            d_cc = donor_cells.get_group(cc_idx).drop(self.critical, axis=1)
            r_cc = r_cc.drop(self.critical, axis=1)
            d_cc_ids = donor_cells.indices[cc_idx]
            r_cc_ids = recip_cells.indices[cc_idx]
            cell_matches = self._match_full(d_cc, r_cc, **kwargs)
            cell_matched_ids = [(r_cc_ids[i], d_cc_ids[j])
                                for i, j in cell_matches]
            matches.extend(cell_matched_ids)
        return matches

    def _match_full(self, donors, recipients, match_args, score_args):
        scores_mat = self._score_records(recipients, donors, **score_args)
        if self.match_method not in MATCH_METHODS:
            raise ValueError(
                "invalid matching method."
                "Possible values are {}".format(",".join(MATCH_METHODS))
            )
        if self.match_method == 'nearest':
            matches = sm.nearest(scores_mat, self.minimize, **match_args)
        elif self.match_method == 'neighbors':
            try:
                k = match_args.pop('k')
                matches = sm.neighbors(
                    scores_mat, k, self.minimize, **match_args
                )
            except KeyError:
                raise ValueError(
                    "neighborhood matching selected but"
                    "missing value for the number of neighbors"
                )
        elif self.match_method == 'hungarian':
            matches = la.hungarian(scores_mat, self.minimize)
        elif self.match_method == 'jonker-volgenant':
            matches = la.jonker_volgenant(scores_mat, self.minimize)
        else:
            raise NotImplementedError(
                "the chosen method is not currently"
                "available: {}".format(self.match_method)
            )
        return matches

    def _score_records(self, x, y, **kwargs):
        # possibly utilize spark here
        metric = self.score_method
        if metric in SKLEARN_SCORE_METHODS:
            if metric == 'manhattan':
                wgt_scl = self.imp_wgts.values
            else:
                wgt_scl = np.sqrt(self.imp_wgts.values)
            return pairwise_distances(
                x*wgt_scl, y*wgt_scl, metric,
                n_jobs=kwargs.get("n_jobs", -1), **kwargs
            )
        return pairwise_distances(
            x, y, metric,
            n_jobs=kwargs.get("n_jobs", -1),
            w=self.imp_wgts.values, **kwargs
        )
