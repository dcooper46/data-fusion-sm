"""
This module offers implicit models to perform data fusion.

Currently only Hot Deck (Statistical Matching) is offered, the
goal being to create a micro-dataset with fused information that
can be used for further analysis.  Macro fusion, estimating the joint
probability distribution of all variables for inference, will be a
future enhancement.
"""
from datafusionsm.implicit_model.hot_deck import HotDeck
from datafusionsm.implicit_model.predictive_mean_matching import PMM
from datafusionsm.implicit_model.predictive_isotonic_matching import PIM

__all__ = ["HotDeck", "PMM", "PIM"]
