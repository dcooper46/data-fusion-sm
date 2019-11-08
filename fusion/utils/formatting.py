"""
various functions to help formatting data
prior to fusion
"""

import pandas as pd


def encode_columns(df, bin_numeric=True):
    """
    extract `categorical` columns and convert to one-hot-encodings.
    keep numeric unbinned columns
    keep columns in original order
    """
    ret_cols = df.columns
    numeric = df.select_dtypes(include='number')
    categorical = df.select_dtypes(include=['object', 'category'])
    if bin_numeric:
        numeric = numeric.apply(bin_numeric_values, axis=0)
    encoded = pd.get_dummies(categorical, columns=categorical.columns)
    joined = encoded.join(numeric)
    selection = []
    for col in ret_cols:
        if col in numeric.columns:
            selection.append(col)
        else:
            encoded_cols = [c for c in encoded.columns if c.startswith(col)]
            for ec in encoded_cols:
                selection.append(ec)
    ret_df = joined[selection]
    return ret_df, ret_df.columns


def bin_numeric_values(x, nbins=10, breaks=None):
    if breaks:
        xmin = x.min()
        xmax = x.max()
        bins = [xmin] + breaks + [xmax]
    else:
        bins = nbins
    return pd.cut(x, bins)
