"""
data fusion via predictive mean matching
"""
import warnings

import pandas as pd
from sklearn.metrics import pairwise_distances

from fusion.implicit_model.base import BaseImplicitModel, ImplicitModelMixin
from fusion.implicit_model import neighbors as nb
from fusion.implicit_model import linear_assignment as la
from fusion.utils.exceptions import NotFittedError
from fusion.utils.model import get_model, train_model, predict


def _build_model(X, y, method, **kwargs):
    k = kwargs.pop("k") if "k" in kwargs else 3
    model = get_model(method, y, **kwargs)
    return train_model(model, X, y, k)


class PMM(ImplicitModelMixin, BaseImplicitModel):
    """

    """
    def __init__(
        self, targets, match_method="nearest", score_method="euclidean",
        model_method="linear"
    ):
        self.targets = targets.split(",")
        self.match_method = match_method
        self.score_method = score_method
        self.model_method = model_method
        self.linking = None
        self.critical = None
        self.results = None

    def fit(self, donors, recipients,
            linking=None, critical=None,
            match_args=None, score_args=None, model_args=None,
            donor_id_col: int = 0, recipient_id_col: int = 0):
        """

        """
        try:
            targets = [(t, donors[t]) for t in self.targets]
        except KeyError:
            raise("requested target not found in donor data")

        kwargs = {"match_args": match_args if match_args else {},
                  "score_args": score_args if score_args else {},
                  "model_args": model_args if model_args else {}}

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

        kwargs["score_args"]["method"] = kwargs["score_args"].get("method", self.score_method)
        kwargs["match_args"]["method"] = kwargs["match_args"].get("method", self.match_method)
        kwargs["model_args"]["method"] = kwargs["model_args"].get("method", self.model_method)

        results = {}
        for target_name, target_data in targets:
            res = {}
            kwargs["model_args"]["target"] = target_data
            _matches, scores, usage = self.get_matches(
                donrs, recips, self.critical, **kwargs
            )
            recip_ids, donor_ids = zip(*_matches)
            matched_donors = donors.iloc[list(donor_ids)].reset_index(drop=True)[donor_id]
            matched_recipients = recipients.iloc[list(recip_ids)].reset_index(drop=True)[recipient_id]

            res["matched_indices"] = pd.DataFrame(_matches, columns=[recipient_id, donor_id])
            res["matches"] = pd.DataFrame(zip(matched_recipients, matched_donors),
                                          columns=[recipient_id, donor_id])
            res["scores"] = scores
            res["usage"] = usage
            results[target_name] = res
        self.results = results
        return self

    def transform(self, donors, recipients):
        """

        """
        if self.results is None:
            raise NotFittedError(
                "This PpcFusion instance is not fitted yet. Call 'fit' with"
                "appropriate arguments before using this method."
            )
        for target, res in self.results.items():
            recip_ids, donor_ids = res["matched_indices"].values.T
            matched_donors = donors.iloc[list(donor_ids)].reset_index(drop=True)
            out_recips = recipients.iloc[list(recip_ids)].reset_index(drop=True)
            ret = out_recips.join(matched_donors[target])

        return ret

    def _match_full(self, donors, recipients, **kwargs):
        model_args = kwargs["model_args"]
        model_method = model_args["method"]
        target = model_args["target"].iloc[donors.index]

        _model_args = {k: v for k, v in model_args.items()
                       if k not in ("method", "target")}
        _target_model = model_args.get("model")

        target_model = (_target_model
                        if _target_model
                        else _build_model(donors, target, model_method, **_model_args))
        donor_preds = predict(target_model, donors)
        recipient_preds = predict(target_model, recipients)

        score_args = kwargs["score_args"]
        score_method = score_args["method"]
        _score_args = {k: v for k, v in score_args.items()
                       if k != "method"}
        scores_mat = self._score_records(
            recipient_preds, donor_preds, score_method, **_score_args
        )

        match_args = kwargs["match_args"]
        match_method = match_args["method"]
        _match_args = {k: v for k, v in match_args.items()
                       if k != "method"}
        if match_method not in self.match_methods:
            raise ValueError(
                "invalid matching method."
                "Possible values are {}".format(",".join(self.match_ma))
            )
        if match_method == 'nearest':
            matches, scores = nb.nearest(scores_mat, True, **_match_args)
        elif match_method == 'neighbors':
            try:
                k = _match_args.pop('k')
                matches, scores = nb.neighbors(
                    scores_mat, k, True, **_match_args
                )
            except KeyError as e:
                warnings.warn(
                    "neighborhood matching selected but "
                    "missing value for the number of neighbors: " + str(e) +
                    "  Using default value of 5"
                )
                matches, scores = nb.neighbors(
                    scores_mat, True, **_match_args
                )
        elif match_method in la.methods:
            matches, scores = la.lap(match_method, scores_mat, True)
        else:
            raise NotImplementedError(
                "the chosen method is not currently"
                "available: {}".format(match_method)
            )
        return matches, scores

    def _score_records(self, x, y, metric, **kwargs):
        x = x if len(x.shape) > 1 else x.reshape(-1, 1)
        y = y if len(y.shape) > 1 else y.reshape(-1, 1)
        return pairwise_distances(
            x, y, metric, n_jobs=kwargs.get("n_jobs", -1), **kwargs
        )
