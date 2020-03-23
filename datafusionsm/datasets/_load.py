""" loaders for sample data """
from os.path import join, dirname

import pandas as pd


def _load_data(data_path, data_file_name):
    """ load data from data_path/data/data_file_name """
    return pd.read_csv(join(data_path, "data", data_file_name), header=0)


def load_tv_panel():
    """ tv panel description """
    cur_dir = dirname(__file__)
    tv_panel = _load_data(cur_dir, "tv-panel.csv")
    return tv_panel


def load_online_survey():
    """ online survey description """
    cur_dir = dirname(__file__)
    online_survey = _load_data(cur_dir, "online-survey.csv")
    return online_survey
