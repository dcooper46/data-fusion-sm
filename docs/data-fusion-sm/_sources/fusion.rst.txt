
.. module:: fusion
   :synopsis: Data fusion in Python

.. currentmodule:: fusion

==============
About *fusion*
==============
The `fusion` package provides common methods to perform data integration
commonly referred to as `data fusion`.  Enriching a data set with external
information can often lead to improvements in modeling accuracy as well as
more insightful analyses.  Having these tools readily available will streamline
this processes and allow data fusion to become a more common task within
data science.


What is Data Fusion
-------------------
Data Fusion is the process of integrating multiple sources of information
about a population collected through some measurement process.  In a marketing
setting, this typically involves combining panels or other population samples
with the goal of enriching the overall behavioral knowledge.  Usually a new,
more robust data source is then created that can be used for more granular
targeting or more accurate predictive modeling.

In general, given data sources A with variables `X, Y` and B with variables `X, Z`
the goal of data fusion is to make inferences on the joint but unobserved data
`X, Y, Z`. The joint density `Y, Z` is inestimable from the observed data and
must be modeled via parameter estimation (maximum likelihood or bayesian
parametric approaches; non-parametric methods) or full synthetic data creation.

To make this estimation feasible, an assumption (often implicit) is undertaken -
the specific variables in either data source, Y and Z, are independent given
the common variables X.  That is, :math:`P(X,Y,Z) = P(Y|X)*P(Z|X)*P(X)`

This assumption is usually not testible and is a source of much research and
debate amongst practitioners.  An oft suggested workaround to loosen this assumption
is the incorporation of auxilary information to inform model priors or uncertainty
of fusion results.  Auxilary information can be difficult to obtain and it is
common practice to assume the condition holds.

Macro vs. Micro
~~~~~~~~~~~~~~~
Data Fusion has often been divided into two main approaches: *macro* and *micro*.

*Macro* fusion is concerned with estimating the probabilistic structure of the
data generating process.  This involves estimating key population parameters
that can be used for inferential statistics; i.e. joint distributions, marginal
distributions, correlations, etc.

*Micro* fusion is aimed at creating a complete synthetic dataset with all of the
desired variables present.  This approach involves imputing values for the
specific variables and the completed dataset used for further modeling and
analysis. The *micro* data fusion approach is the focus of `data-fusion-sm`.


Practical Examples
~~~~~~~~~~~~~~~~~~
* `US Income analysis <https://www.ssa.gov/policy/docs/workingpapers/wp18.pdf>`_

* `Data Fusion in Latin America <http://www.zonalatina.com/Zldata166.htm>`_

* `RSMB Data Integration <https://www.rsmb.co.uk/services/data-integration/>`_

Resources
~~~~~~~~~
* `Statistical matching: a model based approach for data integration <https://ec.europa.eu/eurostat/documents/3888793/5855821/KS-RA-13-020-EN.PDF>`_

* `Data Fusion Through Statistical Matching <http://ebusiness.mit.edu/research/papers/185_Gupta_Data_Fusion.pdf>`_

* `StatMatch - R package <https://cran.r-project.org/web/packages/StatMatch/>`_

* `Statistical Matching: Theory and Practice by Marcello D'Orazio, Marco Di Zio, Mauro Scanu <https://www.amazon.com/Statistical-Matching-Practice-Marcello-DOrazio/dp/0470023538/ref=sr_1_1?keywords=statistical+matching+theory+and+practice&qid=1577824496&sr=8-1>`_

* `Data Fusion at Nielsen <https://www.nielsen.com/wp-content/uploads/sites/3/2019/04/Nielsen-Introduction-to-Data-Fusion-1.pdf>`_

* `Data Fusion at Ipsos <https://www.ipsos.com/sites/default/files/publication/1970-01/IpsosMediaCT_WhitePaper_DataFusion_Jun2011.pdf>`_

* `The design and precision of data-fusion studies <https://www.researchgate.net/publication/287596724_The_design_and_precision_of_data-fusion_studies>`_
