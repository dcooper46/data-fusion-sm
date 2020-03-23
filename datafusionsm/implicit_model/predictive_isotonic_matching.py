"""
data fusion via predictive isotonic matching
"""
import pandas as pd
import numpy as np

from datafusionsm.implicit_model.base import BaseImplicitModel, ImplicitModelMixin
from datafusionsm.utils.exceptions import NotFittedError
from datafusionsm.utils.model import get_model, train_model, predict


def _build_model(X, y, method, **kwargs):
    k = kwargs.pop("k") if "k" in kwargs else 3
    model = get_model(method, y, **kwargs)
    return train_model(model, X, y, k)


class PIM(ImplicitModelMixin, BaseImplicitModel):
    """
    Fuse two data sources together using Predictive Isotonic Matching.
    A model for the target is trained on the donor data and then
    applied to both the donor and recipient data sets.  Statistical Matching
    (hot-deck imputation) is then performed, based on record
    order on the predicted target values (isotonic).  Live values
    (actually observed) from the donor data is then imputed for the recipient.

    Parameters
    ----------
    match_method: str (default='nearest')
        'nearest', 'neighbors', 'hungarian', 'jonker_volgenant'
        algorithm used to match records from each data source

    model_method: str, optional (default None)
        Type/class of model used for predicting the target variable.

    Attributes
    ----------
    critical: array-like[str]
        Critical cells to match within where records must match perfectly

    results: dict[string, array-like[tuple[str, str]]]
        For each target:
            matched id pairs - record indices or id column values
            usage - count of donor usage
            scores - distances for matched records

    Examples
    --------
    >>> from datafusionsm.datasets import load_tv_panel, load_online_survey
    >>> from datafusionsm.implicit_model import PMM
    >>> panel = load_tv_panel()
    >>> survey = load_online_survey()
    >>> pmm = PMM(match_method="jonker_volgenant",
    ...           model_method="forest").fit(panel, survey, critical="age,gender")
    >>> fused = pmm.transform(panel, survey, target="income")
    """

    def __init__(
        self, targets, match_method="unconstrained", model_method=None
    ):
        self.targets = targets.split(",")
        self.match_method = match_method
        self.model_method = model_method
        self.linking = None
        self.critical = None
        self.results = None

    def fit(self, donors, recipients,
            linking=None, critical=None,
            match_args=None, model_args=None,
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

        model_args: dict, optional (default=None)
            Additional arguments for the target model.

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
        try:
            targets = [(t, donors[t]) for t in self.targets]
        except KeyError:
            raise("requested target not found in donor data")

        kwargs = {"match_args": match_args if match_args else {},
                  "model_args": model_args if model_args else {}}

        donor_id = donors.columns[donor_id_col]
        recipient_id = recipients.columns[recipient_id_col]

        donrs, recips, linking = self.check_data(
            donors.drop(donor_id, axis=1),
            recipients.drop(recipient_id, axis=1),
            linking
        )

        self.linking = linking

        if critical:
            self.critical = [l for l in linking
                             if l.rsplit("_", 1)[0] in critical]
        else:
            self.critical = []

        kwargs["model_args"]["method"] = kwargs["model_args"].get("method", self.model_method)
        kwargs["match_args"]["method"] = kwargs["match_args"].get("method", self.match_method)

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
        Using fused ids, impute information from donor data to the
        recipient data.

        Parameters
        ----------
        donors: pandas.DataFrame
            Records containing information to be donated

        recipients: pandas.DataFrame
            Records that will receive information

        Returns
        -------
        ret: pandas.DataFrame
            New DataFrame containing dontated information
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

        # no method set and no model passed
        if not (model_method or _target_model):
            self.model_method = "linear"

        target_model = (_target_model
                        if _target_model
                        else _build_model(donors, target, self.model_method, **_model_args))
        donor_preds = predict(target_model, donors)
        recipient_preds = predict(target_model, recipients)

        match_args = kwargs["match_args"]
        match_method = match_args["method"]
        _match_args = {k: v for k, v in match_args.items()
                       if k != "method"}

        matches, scores = isotonic_matching(
            recipient_preds,
            donor_preds,
            match_method,
            **_match_args
        )

        return matches, scores


def isotonic_matching(x, y, method='unconstrained', n_bins=None):
    """ perform isotonic matching """
    n_bins = n_bins if n_bins else 4

    assert len(x.shape) == len(y.shape)

    if len(x.shape) > 1:
        assert x.shape[1] == y.shape[1]

        scale = [10**i for i in range(x.shape[1])]
        x = x.dot(scale)
        y = y.dot(scale)

    steps = np.arange(0, 1, round(1 / n_bins, 2))
    x_quants = [np.quantile(x, q) for q in steps]
    y_quants = [np.quantile(y, q) for q in steps]

    x_binned = np.digitize(x, x_quants, right=True)
    y_binned = np.digitize(y, y_quants, right=True)

    x_ids = np.array(list(enumerate(x)))
    y_ids = np.array(list(enumerate(y)))

    if method == "unconstrained":
        matches, scores = unconstrained_iso(x_ids, y_ids, x_binned, y_binned, n_bins)
    else:
        raise NotImplementedError("constrained isotonic matching not yet implemented")

    return matches, scores


def unconstrained_iso(x_ids, y_ids, x_binned, y_binned, n_bins):
    matches = []
    scores = []
    overflow = []
    underflow = []
    for quantile in range(n_bins + 1):
        x_quants = np.where(x_binned == quantile)[0].astype(int)
        y_quants = np.where(y_binned == quantile)[0].astype(int)

        if x_quants.size == 0:
            if y_quants.size > 0:
                underflow = y_quants
            continue
        if y_quants.size == 0:
            overflow = x_quants
            continue
        x_quants = np.concatenate((x_quants, overflow)).astype(int)
        xq_ids, xq = zip(*x_ids[x_quants])
        yq_ids, yq = zip(*y_ids[y_quants])
        sorted(xq_ids)
        sorted(yq_ids)
        for i, xid in enumerate(xq_ids):
            matches.append((int(xid), int(yq_ids[i % len(yq_ids)])))
            scores.append(abs(xq[i] - yq[i % len(yq_ids)]))
        overflow = []

    if len(overflow) > 0:
        xq_ids, xq = zip(*x_ids[overflow])
        yq_ids, yq = zip(*y_ids[underflow])
        sorted(xq_ids)
        sorted(yq_ids)
        for i, xid in enumerate(xq_ids):
            matches.append((int(xid), int(yq_ids[i % len(yq_ids)])))
            scores.append(abs(xq[i] - yq[i % len(yq_ids)]))

    return matches, scores
