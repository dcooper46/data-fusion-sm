Evalute Matches
===============

Data fusion / statistical matching results are most appropriately used at an
aggregate level, as there is no way to `guarantee` the donated values would
have been observed for the individual records.  Thus, donated data is usually
either analyzed at an aggregate level for inferential statistics or parameter
estimates, or used in subsequent modeling where the reality of the individual
value is less important.  However, its still often desired to have realistic
record matches to justify the donated values and increase confidence in
insights derived from them.

The easiest way to evaluate individual matches is to consider the common
variables (those used in the matching) for the matched pairs.  How well these
variables align gives an indication of the accuracy of the match.  This is
esspecially important when using nearest neighbor methods.  Agreement then
reinforces the assumption that records that "look alike" will "act alike",
justifying the donated variable.

.. autofunction:: datafusionsm.evaluation.match_accuracy
