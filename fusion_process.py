# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 07:46:23 2017

unconstrained distance hot-deck method

@author: dancoope
"""
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from scipy.optimize import linear_sum_assignment
from time import time


def load_data(files):
    donors = pd.read_csv(files.get('donors'))
    recips = pd.read_csv(files.get('recipients'))
    maps = np.load(files.get('maps')).item()

    return donors, recips, maps


def to_category(col):
    return col.astype('category')


def bin_numeric(x):
    """
    """
    binned = np.digitize(x, np.histogram(x, bins='fd')[1])

    return pd.Series(binned)


def get_counts(x):
    """
        return counts by distinct value given a pandas series
    """
    return x.groupby(x).size()


def get_probs(x):
    """
        given an np array or pandas series, return
        the probability distribution
    """
    return x / x.sum()


def gini(x):
    """
        return gini index of variable

        Assumptions:
            - continuous variables are already properly binned
            - x is a pandas series
    """
    if x.dtype.name != 'category':
        x = bin_numeric(x)

    counts = get_counts(x)
    p = get_probs(counts)

    return 1 - (p**2).sum()


def entropy(x):
    """
        return entropy of variable

        Assumptions:
            - continuous variables are already properly binned
            - x is a pandas series

        ? return max entropy as comparison ?
    """
    if x.dtype.name != 'category':
        x = bin_numeric(x)

    counts = get_counts(x)
    p = get_probs(counts)

    return -(p * np.log2(p)).sum()


def scale(arr):
    if arr.ndim < 2:
        return np.nan_to_num((arr - arr.mean()) / arr.std())
    return np.nan_to_num(
            (arr - arr.mean(axis=1, keepdims=True))
            / arr.std(axis=1, keepdims=True)
        )


def score(dist, p=2):
    return np.power(np.mean(np.sign(dist)*np.power(abs(dist), p)), p)


def wgtd_score(dist, wgts, p=2):
    return np.power(abs(sum(wgts * np.sign(dist) * np.power(abs(dist), p))), p)


def grange(x):
    return np.max(x) - np.min(x)


def gower(x, y, wgts):
    def cat_equal(x, y):
        return np.not_equal(x, y)

    def cat_dist(x, y, d):
        return abs(x - y) / d  # range of education values

    d = cat_equal(x, y).astype(float)
    d.put(2, cat_dist(x[2], y[2], grange(np.concatenate((x, y)))))

    return (d * wgts).sum() / wgts.sum()


def importance_wgts(data, method='entropy'):
    if method == 'entropy':
        data = data.apply(to_category)
        return np.array([entropy(data[var]) for var in data.columns])
    else:
        return np.ones((1, data.shape[1]))


def get_fusion_vars(indices, donors, fvars, method='mode'):
    matches = donors.take(indices)[fvars]
    if method == 'mode':
        return matches.mode().mean().astype(int)
    elif method == 'random':
        return matches.take(np.random.randint(matches.shape[0]), axis=0)
    else:
        raise NotImplementedError


def get_matches(distances, method, **kwargs):
    if method == 'knn':
        return distances.argsort(axis=1)[:, :k]
    else:
        raise NotImplementedError


def fuse(donors, recipients, linking, fusion, critical=None, imp_wgts=None,
         matching='knn', donate_by='mode', njobs=2, k=10):
    if critical is None:
        if imp_wgts:
            # try to group by variable with highest weight
            critical = linking[imp_wgts.argmax()]
        else:
            # take variable with most unique values to ensure smallest groups
            critical = linking[np.argmax(
                [recipients[var].unique().size for var in linking]
            )]

    cc_donors = donors.groupby(critical, sort=False)
    cc_recips = recipients.groupby(critical, sort=False)

    if imp_wgts is None:
        imp_wgts = np.ones(len(linking))

    # can use custom distance or standard like Gower (handles mixed types)
    fused_data = {}
    t0 = time()
    for name, recips_cc in cc_recips:
        t1 = time()
        donors_cc = cc_donors.get_group(name)
        dist = pairwise_distances(recips_cc[linking], donors_cc[linking],
                                  gower, n_jobs=njobs, wgts=imp_wgts)
        top_matches = get_matches(dist, matching, k=k)
        fused_vars = np.apply_along_axis(
            get_fusion_vars, 1, top_matches, donors_cc, fusion, donate_by
        )
        fused_vars = pd.DataFrame(fused_vars).set_index(recips_cc.index)
        fused_vars.columns = fusion
        fused_data[name] = recips_cc.join(fused_vars, how='right')
        print("finished cell {} in {} seconds".format(name, time() - t1))
    print("finished matching in {} seconds".format(time() - t0))

    fused_df = fused_data[1]
    fused_df = fused_df.append(
        [df for key, df in fused_data.items() if key != 1]
    )

    return fused_df


def main(files, vars_, matching='knn', k=10, donate_by='mode'):
    donors, recipients, maps = load_data(files)

    linking_vars, fusion_vars, critical_vars = map(
        vars_.get, ['linking_vars', 'fusion_vars', 'critical_vars']
    )

    imp_wgts = importance_wgts(recipients.append(donors)[linking_vars])

    fused_data = fuse(donors, recipients, linking_vars, fusion_vars,
                      critical_vars, imp_wgts, matching=matching,
                      donate_by=donate_by, k=k)

    fused_data.to_csv("fused_data_{}nn-rand.csv".format(k), index=False)


if __name__ == "__main__":

    files = dict()
    files['donors'] = 'donors.csv'
    files['recipients'] = 'recipients.csv'
    files['maps'] = 'maps.npy'

    vars_ = dict(critical_vars=['age_enc'],
                 linking_vars=['job_enc', 'marital_enc', 'education_enc',
                               'default_enc', 'housing_enc', 'loan_enc'],
                 target=['y'],
                 fusion_vars=[
                    'contact_enc', 'month_enc', 'day_of_week_enc',
                    'campaign_enc', 'pdays_enc', 'previous_enc', 'poutcome_enc'
                 ])
    match_mthd = 'knn'
    donate_mthd = 'random'
    k = 10

    main(files, vars_, matching=match_mthd, donate_by=donate_mthd, k=k)

# donated = []
# for idx, row in recipients.iterrows():
#    cc = cc_donors.get_group(row.age_enc)
#    rr = row[linking_vars].values
#    distances = np.apply_along_axis(
#           gower, 1, cc[linking_vars].values, rr, imp_wgts)
#    matches = distances.argsort()[:k]
#
#    # take mode of fused variables for matches
#    donated_vars = cc.take(matches)[fusion_vars].mode()
#    donated.append([row.uid] + donated_vars.values)

# ***************
# **test**
"""
cc_recips1 = cc_recips.get_group(1)
cc_donors1 = cc_donors.get_group(1)
dist = pairwise_distances(cc_recips1[linking_vars], cc_donors1[linking_vars], gower, wgts=imp_wgts)
top_matches = dist.argsort(axis=1)[:,:k]
fused_vars = np.apply_along_axis(get_fusion_vars, 1, top_matches, cc_donors1, fusion_vars)
fused_vars = pd.DataFrame(fused_vars)
fused_vars.set_index(cc_recips1.index, inplace=True)
fused_vars.columns = fusion_vars
merged = cc_recips1.join(fused_vars, how='right')
merged.head()


r0 = recipients[:1].iloc[0]
cc = cc_donors.get_group(r0.age_enc)
cc_scaled = scale(cc[linking_vars].values)
r0_scaled = scale(r0[linking_vars].values)

distances = abs(cc_scaled - r0_scaled) * imp_wgts
d_matches = distances.sum(axis=1).argsort()

# or

# gower metric preserves data types (no scaling)
cc0 = cc[linking_vars].values
r00 = r0[linking_vars].values

distances1 = np.apply_along_axis(gower, 1, cc0, r00, imp_wgts)
d_matches1 = distances1.argsort()[:k]

# or

distances2 = prws.euclidean_distances(cc_scaled, r0_scaled.reshape(1, -1)).reshape(cc.shape[0],)
d2_matches = distances2.argsort()
"""