"""Computation of the dissimilarity representation of a dataset of
objects from a set of k prototypes (landmarks), given a distance
function. The result is a Euclidean embedding of the dataset with k
dimensions.

See Olivetti E., Nguyen T.B., Garyfallidis, E., The Approximation of
the Dissimilarity Projection, http://dx.doi.org/10.1109/PRNI.2012.13

"""

import numpy as np
from subsampling import compute_subset


def compute_dissimilarity(dataset, distance, k,
                          prototype_policy='sff', verbose=False):
    """Compute the dissimilarity (distance) matrix between a dataset of N
    objects and prototypes, where prototypes are selected among the
    objects with a given policy.

    Parameters
    ----------
    dataset : list or array of objects
           an iterable of objects.
    distance : function
           Distance function between groups of objects or sets of objects.
    k : int
           The number of prototypes/landmarks.
    prototype_policy : string
           The prototype selection policy. The default value is 'sff',
           which is highly scalable.
    verbose : bool
           If true prints some messages. Deafault is True.

    Return
    ------
    dissimilarity_matrix : array (N, k)

    See Also
    --------
    subsampling.furthest_first_traversal,
    subsampling.subset_furthest_first

    Notes
    -----

    """
    if verbose:
        print("Generating %s prototypes with policy %s." % (k, prototype_policy))

    prototype_idx = compute_subset(dataset, distance, k,
                                   landmark_policy=prototype_policy)
    prototypes = [dataset[i] for i in prototype_idx]
    dissimilarity_matrix = distance(dataset, prototypes)
    return dissimilarity_matrix, prototype_idx
