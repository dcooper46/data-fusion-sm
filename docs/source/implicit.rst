.. module:: implicit
   :synopsis: Joining custom data sources via implicit fusion

.. currentmodule:: datafusionsm.implicit_model

Implicit Fusion
===============


Implicit Fusion refers to joining data sources using information implicit
within each.  Under one framework, Statistical-Matching (Hot Deck Imputation),
overlapping features are used to determined how simlilar records are, and
assignments/matches are made using various algorithms.  Under another method,
a model is first built to predict the target(s).  Predictions are made for both
data sources, and they are matched based on these.  This method is sometimes
referred to as Predictive Mean Matching. These are the core implicit algorithms
most often used in the data fusion literature and applications, and are the
only ones currently offered.


Implicit Models
^^^^^^^^^^^^^^^

.. toctree::
    :maxdepth: 1

    hot_deck
    pmm
