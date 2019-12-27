""" Base classes for implicit_models """
from collections import Counter

from datafusionsm.utils.formatting import encode_columns


class BaseImplicitModel:
    """ Base implicit_model """
    match_methods = ["nearest", "neighbors", "hungarian", "jonker_volgenant"]
    local_score_methods = {"gower": None, "exact": None}
    sklearn_score_methods = {"cosine", "euclidean", "manhattan"}
    importance_methods = ["entropy", "gini", "info_gain", "linear", "tree"]

    def fit(self, donors, recipients, *args, **kwargs):
        pass

    def transform(self, donors, recipients, *args, **kwargs):
        pass

    def fit_transform(self, donors, recipients, *args, **kwargs):
        self.fit(donors, recipients, *args, **kwargs)
        return self.transform(donors, recipients, *args, **kwargs)


class ImplicitModelMixin:
    """ mixin methods for ImplicitModels """

    def check_data(self, donors, recipients, score_method, linking=None):
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

    def get_matches(self, donors, recipients, critical, **kwargs):
        if critical is None or critical == []:
            donors["groupby"] = -1
            recipients["groupby"] = -1
            critical = "groupby"
        matches, scores = self._match_critical(
            donors, recipients, critical, **kwargs)
        _, matched_donors = zip(*matches)
        usage = Counter(matched_donors)
        return matches, scores, usage

    def _match_critical(self, donors, recipients, critical, **kwargs):
        donor_cells = donors.groupby(critical, sort=False)
        recip_cells = recipients.groupby(critical, sort=False)
        matches = []
        scores = []
        for cc_idx, r_cc in recip_cells:
            d_cc = donor_cells.get_group(cc_idx).drop(critical, axis=1)
            r_cc = r_cc.drop(critical, axis=1)
            d_cc_ids = donor_cells.indices[cc_idx]
            r_cc_ids = recip_cells.indices[cc_idx]
            cell_matches, cell_scores = self._match_full(d_cc, r_cc, **kwargs)
            cell_matched_ids = [(r_cc_ids[i], d_cc_ids[j])
                                for i, j in cell_matches]
            matches.extend(cell_matched_ids)
            scores.extend(cell_scores)
        return matches, scores

    def _match_full(self, donors, recipients, **kwargs):
        # must be overwritten by inheriting class
        pass

    def _score_records(self, x, y, metric, **kwargs):
        pass
