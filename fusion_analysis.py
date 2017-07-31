# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 09:28:25 2017

@author: dancoope
"""

import numpy as np
import pandas as pd


def load_data(files):
    fused = pd.read_csv(**files.get('fused'))
    raw_data = pd.read_csv(**files.get('raw'))
    recips_orig = raw_data.take(fused['uid'])
    maps = np.load(files.get('maps')).item()

    return fused, raw_data, recips_orig, maps


def encode(data, vars_, maps):
    for var in vars_:
        data[var+'_enc'] = data[var].apply(lambda x: maps.get(var)[str(x)])

    return data


def evaluate(orig, fused, linking_vars, fusion_vars):
    linked_correct, fused_correct = {}, {}
    n = orig.shape[0]
    for var in linking_vars + fusion_vars:
        if var.contains('_enc'):
            enc_var = var
            var = var.strip('_enc')
        else:
            enc_var = var + '_enc'
        n_correct = (orig[enc_var].reset_index() == fused[enc_var].reset_index()).sum()[enc_var]
        correct = (n_correct, n_correct / n)
        if var in linking_vars:
            linked_correct[var] = correct
        else:
            fused_correct[var] = correct

    return linked_correct, fused_correct


def main(files, vars_):

    fused, raw, recips_orig, maps = load_data(files)
    linking_vars, fusion_vars = map(vars_.get, ('linking', 'fusion'))

    raw = encode(raw, linking_vars, maps)

    linked_correct, fused_correct = evaluate(recips_orig, fused, linking_vars, fusion_vars)

    print(linked_correct)
    print(fused_correct)


if __name__ == "__main__":

    fusion_params = dict(filepath_or_buffer="fused_data_10nn-mode.csv", sep=",")
    raw_data_params = dict(filepath_or_buffer="bank-additional-full.csv", sep=";")

    linking_vars = ['job', 'marital', 'education', 'default', 'housing', 'loan']
    critical_cells = ['age']  # also a linking var, but must match so no need to evaluate
    target = ['y']
    fusion_vars = ['contact', 'month', 'day_of_week', 'campaign', 'pdays', 'previous', 'poutcome']

    files = {'fusion': fusion_params, 'raw': raw_data_params}
    vars_ = {'linking': linking_vars,
             'critical': critical_cells,
             'target': target,
             'fusion': fusion_vars}

    main(files, vars_)
