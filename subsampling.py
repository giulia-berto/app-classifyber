"""Functions to get a subset of objects that represents the initial
dataset. The objects in such subset are sometimes called landmarks or
prototypes.

"""

import numpy as np


def furthest_first_traversal(dataset, k, distance, permutation=True):
    """This is the farthest first traversal (fft) algorithm which selects
    k objects out of an array of objects (dataset). This algorithms is
    known to be a good sub-optimal solution to the k-center problem,
    i.e. the k objects are sequentially selected in order to be far
    away from each other.

    Parameters
    ----------

    dataset : array of objects
        an iterable of objects which supports advanced indexing.
    k : int
        the number of objects to select.
    distance : function
        a distance function between two objects or groups of objects,
        that given two groups as input returns the distance or distance
        matrix.
    permutation : bool
        True if you want to shuffle the objects first. No
        side-effect on the input dataset.

    Return
    ------
    idx : array of int
        an array of k indices of the k selected objects.

    Notes
    -----
    - Hochbaum, Dorit S. and Shmoys, David B., A Best Possible
    Heuristic for the k-Center Problem, Mathematics of Operations
    Research, 1985.
    - http://en.wikipedia.org/wiki/Metric_k-center

    See Also
    --------
    subset_furthest_first

    """
    if permutation:
        idx = np.random.permutation(len(dataset))
        dataset = dataset[idx]
    else:
        idx = np.arange(len(dataset), dtype=np.int)

    T = [0]
    while len(T) < k:
        z = distance(dataset, dataset[T]).min(1).argmax()
        T.append(z)

    return idx[T]


def subset_furthest_first(dataset, k, distance, permutation=True, c=2.0):
    """The subset furthest first (sff) algorithm is a stochastic
    version of the furthest first traversal (fft) algorithm. Sff
    scales well on large set of objects (dataset) because it
    does not depend on len(dataset) but only on k.

    Parameters
    ----------

    dataset : list or array of objects
        an iterable of objects.
    k : int
        the number of objects to select.
    distance : function
        a distance function between groups of objects, that given two
        groups as input returns the distance matrix.
    permutation : bool
        True if you want to shuffle the objects first. No
        side-effect.
    c : float
        Parameter to tune the probability that the random subset of
        objects is sufficiently representive of dataset. Typically
        2.0-3.0.

    Return
    ------
    idx : array of int
        an array of k indices of the k selected objects.

    See Also
    --------
    furthest_first_traversal

    Notes
    -----
    See: E. Olivetti, T.B. Nguyen, E. Garyfallidis, The Approximation
    of the Dissimilarity Projection, Proceedings of the 2012
    International Workshop on Pattern Recognition in NeuroImaging
    (PRNI), pp.85,88, 2-4 July 2012 doi:10.1109/PRNI.2012.13
    """
    size = compute_subsample_size(k, c=c)
    if permutation:
        idx = np.random.permutation(len(dataset))[:size]
    else:
        idx = range(size)

    return idx[furthest_first_traversal(dataset[idx],
                                        k, distance,
                                        permutation=False)]


def compute_subsample_size(n_clusters, c=2.0):
    """Compute a subsample size that takes into account a possible cluster
    structure of the dataset, in n_clusters, based on a solution of
    the coupon collector's problem, i.e. k*log(k).

    Notes
    -----
    See: E. Olivetti, T.B. Nguyen, E. Garyfallidis, The Approximation
    of the Dissimilarity Projection, Proceedings of the 2012
    International Workshop on Pattern Recognition in NeuroImaging
    (PRNI), pp.85,88, 2-4 July 2012 doi:10.1109/PRNI.2012.13

    """
    return int(max(1, np.ceil(c * n_clusters * np.log(n_clusters))))



def compute_subset(dataset, distance, num_landmarks,
                   landmark_policy='sff'):
    """Wrapper code to dispatch the computation of the subset according to
    the required policy.
    """
    if landmark_policy == 'random':
        landmark_idx = np.random.permutation(len(dataset))[:num_landmarks]
    elif landmark_policy in ('fft', 'minmax'):
        landmark_idx = furthest_first_traversal(dataset,
                                                 num_landmarks, distance)
    elif landmark_policy == 'sff':
        landmark_idx = subset_furthest_first(dataset, num_landmarks, distance)
    else:
        if verbose:
            print("Landmark selection policy not supported: %s" % landmark_policy)

        raise Exception

    return landmark_idx


