import numpy as np
from scipy.spatial import distance_matrix

try:
    from joblib import Parallel, delayed, cpu_count
    joblib_available = True
except:
    joblib_available = False

print("joblib_available = %s" %joblib_available)
if joblib_available:
	print("nr cpu = %s" %cpu_count())

def euclidean_distance(A, B):
    """Wrapper of the euclidean distance between two vectors, or array and
    vector, or two arrays.
    """
    return distance_matrix(np.atleast_2d(A), np.atleast_2d(B), p=2)


def parallel_distance_computation(A, B, distance, n_jobs=-1,
                                  granularity=2, verbose=False,
                                  job_size_min=1000):
    """Computes the distance matrix between all objects in A and all
    objects in B in parallel over all cores.

    This function can be partially instantiated with a given distance,
    in order to obtain a the parallel version of a distance function
    with the same signature as the distance function. Example:
    distance_parallel = functools.partial(parallel_distance_computation, distance=distance)
    """
    if (len(A) > job_size_min) and joblib_available and (n_jobs != 1):
        if n_jobs is None or n_jobs == -1:
            n_jobs = cpu_count()

        if verbose:
            print("Parallel computation of the distance matrix: %s cpus." % n_jobs)

        if n_jobs > 1:
            tmp = np.linspace(0, len(A), granularity * n_jobs + 1).astype(np.int)
        else:  # corner case: joblib detected 1 cpu only.
            tmp = (0, len(A))

        chunks = zip(tmp[:-1], tmp[1:])
        dissimilarity_matrix = np.vstack(Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(distance)(A[start:stop], B) for start, stop in chunks))
    else:
        dissimilarity_matrix = distance(A, B)

    if verbose:
        print("Done.")

    return dissimilarity_matrix
