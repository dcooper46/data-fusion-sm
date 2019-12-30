
Welcome to fusion's documentation!
==================================

:doc:`fusion <fusion>` is a Python package that provides classes and methods to
join two data sources that contain disparate information.  The goal is to share
information that exists in only one data source.  This is done typically by
matching the most similar records together; the assumption being that simliar
records or people will behave similarly for the chosen measurable.

Two modes are offered to perform fusion: `custom` and `ppc`.  Within each of
these modes, two types of fusion are offered: `implicit` and `explicit`.

:doc:`implicit <implicit>` fusion refers to the general use case of joining
multiple data sources. These can be in-house or 3rd party panels and surveys.
Various methods to perform the fusion are available, though currently only for
implicit fusion.  The main method offered for implicit fusion is
`Statistical Matching`, whereby the closeness of records are determined by
some similarity score or metric.  Matches are then assigned based on this.

`Implicit` fusion will utilize information inherent in both data sources in
order to determine a relationship between the two.  This typically means
finding similar records based on some criteria.  `Explicit` fusion, however,
will utilize information and patterns from the donating data source and apply
that knowledge to the receving data source.  This usually involves creating
a predictive model based on the donating data and applying it on the receiving
data.

Minimal Examples
----------------

Fusion of two surveys via Hot-Deck imputation::

    import pandas as pd
    from fusion.implicit_model import HotDeck

    survey1 = pd.read_csv("survey1")
    survey2 = pd.read_csv("survey2")

    impF = HotDeck()
    fused_survey = impF.fit_transform(survey1, survey2)

Fusion of two surveys via Predictive Mean Matching::

    import pandas as pd
    from fusion.implicit_model import PMM

    survey1 = pd.read_csv("survey1")
    survey2 = pd.read_csv("survey2")

    impF = PMM("forest")
    fused_survey = impF.fit_transform(survey1, survey2)



Contents
--------
.. toctree::
   :maxdepth: 1

   implicit
   evaluation
   exploration



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
