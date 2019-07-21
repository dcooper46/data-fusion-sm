import pandas as pd
import numpy as np
from itertools import permutations


def _safe_divide(n, d):
    if (isinstance(n, (np.ndarray, pd.Series)) or
            isinstance(d, (np.ndarray, pd.Series))):
        return np.array([_safe_divide(x, y) for x, y in zip(n, d)])
    elif isinstance(n, list) or isinstance(d, list):
        return _safe_divide(np.array(n), np.array(d))
    else:
        return n / d if d != 0 else 1


def _join_panels(matches, donors, recipients):
    matches_df = pd.DataFrame(matches, columns=["recipient_id", "donor_id"])
    matched_recipients = (
        recipients
        .rename(columns={"panel_id": "recipient_id"})
        .merge(matches_df, on="recipient_id")
    )
    return matched_recipients.merge(
        donors.rename(columns={"panel_id": "donor_id"}),
        on="donor_id",
        suffixes=("_r", "_d")
    )


def _permute_values(panel, feats):
    unique_values = dict()
    perms = dict()
    for feat in feats:
        values = panel[feat].unique()
        values = [v for v in values if v != 'Unknown']
        unique_values[feat] = values
        feat_perms = list(permutations(values, 2))
        feat_perms.extend((v, v) for v in values)
        perms[feat] = feat_perms

    return unique_values, perms


def _get_fused_counts(joined_panels, feats):
    # count of each (panel, fused) label combination for each feature
    matches = dict()
    for feat in feats:
        matches[feat] = (
            joined_panels
            .groupby([feat + "_r", feat + "_d"])
            .size()
            .to_dict()
        )
    return matches


def _create_result_set(perms, feats):
    results = pd.DataFrame(columns=['genre', 'recipient', 'donor', 'count'])
    for feat in feats:
        data = [(feat, x1, x2, 0) for x1, x2 in perms[feat]]
        results = results.append(
            pd.DataFrame(data, columns=['genre', 'recipient', 'donor', 'count'])
        )
    results.reset_index(inplace=True, drop=True)
    return results


def demo_accuracy(matches, donors, recipients, feats):
    """
    Calculate and display the accuracy of the expected demographics
    post-fusion.

    Parameters
    ----------
    matches: array-like
        list of matching id pairs resulting from fusion

    donors: pandas.DataFrame
        donating records

    recipients: pandas.DataFrame
        receiving records

    feats: array-like
        what demographic features were fused on and should be evaluated

    Returns
    -------
    str
        formatted output string containing demographic evaluation
    results: pandas.DataFrame
        detailed counts of demographic matches
    """

    fused_demos = _join_panels(matches, donors, recipients)

    unique_values, perms = _permute_values(donors, feats)
    matched_demo_counts = _get_fused_counts(fused_demos, feats)
    results = _create_result_set(perms, feats)

    out_buffer = []
    for feat in feats:
        out_buffer.append(feat + ":\n")

        uvals = sorted(unique_values[feat])

        counts = pd.DataFrame(index=uvals, columns=uvals)

        for perm in perms[feat]:
            row, col = perm
            idx = results.loc[(results['recipient'] == row) &
                              (results['donor'] == col)].index
            cnt = matched_demo_counts[feat].get((row, col), 0)
            counts[row][col] = cnt
            results.loc[idx, 'count'] = cnt

        N = counts.values.sum()
        if N == 0:
            print(f"Divide by 0!  {feat}")
            print(counts)
            continue
        acc = np.round(np.trace(counts) / N, 4)

        panel_cnts = counts.sum(axis=1).apply(int)
        counts = counts.sort_index(0).sort_index(1)
        counts['recipient'] = panel_cnts

        pred_cnts = counts.sum(axis=0).apply(int)
        counts.loc['donor'] = pred_cnts

        out_buffer.append(counts.to_string())
        out_buffer.append("\n")

        out_buffer.append(f"\tacc:\t{acc}\n")

        num = counts.loc['donor'][uvals]
        div = counts['recipient'].loc[uvals]
        rec = np.round(_safe_divide(num, div), 4)

        rec = [str(r) for r in rec]

        out_buffer.append('\tdonor/recip\t' + '\t'.join(uvals))
        out_buffer.append('\n')
        out_buffer.append('\t\t\t' + '\t'.join(rec))
        out_buffer.append('\n\n')

    return '\n'.join(out_buffer), results
