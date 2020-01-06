.. _intro:

=================
Fusion Evaluation
=================


Fusion results aren't easily evaluated like other machine learning
methods, as there is not usually labeled training data to measure
accuracy or other common metrics.

However, we do have available the features used to match the records.
As we want the matches to be as accurate as possible, evaluating how
these values compare for the matched records can give an good indication of how
reasonable the fusion is.

We can also look at how well donated target variables retain their
distributions.  Recall, a main goal of data fusion is to donate information
and treat it as though it were measured.  Since the datasets being fused
represent the same population, the donated data should resemble as
closely as possible the original data that was measured.  Comparing the
target variable distributions pre and post fusion can give an indication
of the fusion's effectiveness.

.. toctree::
   :maxdepth: 1

   distributions
   matches

:doc:`Home <../index>`
