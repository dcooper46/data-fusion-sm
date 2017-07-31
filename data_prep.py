# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 16:06:52 2017

@author: dancoope
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def age_group(age):
    if age < 100:
        return age // 10
    else:
        return 9


def education_levels(edu):
    if edu == 'unknown':
        return 0
    elif edu == 'illiterate':
        return 1
    elif edu == 'basic.4y':
        return 2
    elif edu == 'basic.6y':
        return 3
    elif edu == 'basic.9y':
        return 4
    elif edu == 'high.school':
        return 5
    elif edu == 'professional.course':
        return 6
    elif edu == 'university.degree':
        return 7
    else:
        raise ValueError


def encode(col):
    le = LabelEncoder()
    le.fit(col)
    return le.transform(col), le


def encode_variables(data, vars_):
    maps = {}
    # preserve ordinal variables like education and age
    data['age_enc'] = data['age'].apply(age_group)
    data['education_enc'] = data['education'].apply(education_levels)
    for var in vars_:
        if var in ['age', 'education']:
            continue
        data[var+'_enc'], model = encode(data[var])
        maps[var] = {str(model.inverse_transform(c)): c for c in range(len(model.classes_))}

    maps['age'] = {'0-9': 0,
                   '10-19': 1,
                   '20-29': 2,
                   '30-39': 3,
                   '40-49': 4,
                   '50-59': 5,
                   '60-69': 6,
                   '70-79': 7,
                   '80-89': 8,
                   '90+': 9}
    maps['education'] = {'unknown': 0,
                         'illiterate': 1,
                         'basic.4y': 2,
                         'basic.6y': 3,
                         'basic.9y': 4,
                         'high.school': 5,
                         'professional.course': 6,
                         'university.degree': 7}

    return data, maps


def split_data(data, vars_, donor_perc=0.45, seed=46):
    donors = data[vars_].sample(frac=donor_perc, seed=seed)

    recip_idx = data.index.difference(donors.index)
    recipients = data[vars_].take(recip_idx)

    return donors, recipients


def main(data_file, linking_vars, fusion_vars, target, *args, **kwargs):

    sep = kwargs.get('sep', ',')
    data = pd.read_csv(data_file, sep=sep)
    data['uid'] = data.index

    data, maps = encode_variables(data, linking_vars+fusion_vars)

    linking_vars = [var+'_enc' for var in linking_vars]
    fusion_vars = [var+'_enc' for var in fusion_vars]
    keeps = ['uid'] + linking_vars + fusion_vars + target
    donors, recipients = split_data(data, keeps)

    donors.to_csv("donors.csv", index=False)
    recipients.to_csv("recipients.csv", index=False)
    np.save("maps", maps)


if __name__ == "__main__":
    data_file = "bank-additional-full.csv"
    linking_vars = ['age', 'job', 'marital', 'education', 'default', 'housing', 'loan']
    target = ['y']
    fusion_vars = ['contact', 'month', 'day_of_week', 'campaign', 'pdays', 'previous', 'poutcome']

    main(data_file, linking_vars, fusion_vars, target, sep=";")
