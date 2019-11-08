"""
This module offers implicit models to perform data fusion.

Currently only Hot Deck (Statistical Matching) is offered, the
goal being to create a micro-dataset with fused information that
can be used for further analysis.  Macro fusion, estimating the joint
probability distribution of all variables for inference, will be a
future enhancement.
"""
from fusion.implicit.hot_deck import HotDeck

__all__ = ["HotDeck"]
