
****************************************************
data-fusion-sm: data fusion via statistical matching
****************************************************

:doc:`data-fusion-sm <fusion>` is a Python package that provides classes and
methods to join two data sources that contain disparate information; the goal
is to share information that is specific to either data source.  This is often
referred to as `data fusion` or `statistical matching`.

`Data fusion` should be understood as a much broader class of algorithms that
can include techniques designed for various types of data including sensors,
medical equipment, genomes, surveys, and many more.  Algorithms for general
missing data imputation can also be used to perform data fusion, irrespective
of data type.  `Statistical matching` is a specific set of methods designed for
data similar to surveys and panels, where records are typically people with a
set of measured attributes.  `data-fusion-sm` is designed to work with such
data and includes common statistical matching methods to perform data fusion.



Minimal Examples
----------------

Fusion of two surveys via Hot-Deck imputation::

    import pandas as pd
    from datafusionsm.implicit_model import HotDeck

    survey1 = pd.read_csv("survey1")
    survey2 = pd.read_csv("survey2")

    impF = HotDeck()
    fused_survey = impF.fit_transform(survey1, survey2)

Fusion of two surveys via Predictive Mean Matching::

    import pandas as pd
    from datafusionsm.implicit_model import PMM

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

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
