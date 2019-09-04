"""Streamline and bundle distances based on the Euclidean (flip)
distance between endpoints.
"""

from __future__ import division, print_function, absolute_import
from numpy.linalg import norm
from dipy.tracking.distances import bundles_distances_mdf
from dipy.align.bundlemin import distance_matrix_mdf
import numpy as np


def streamline_distance_endpoints(s_A, s_B):
    """Streamline distance based on Euclidean (flip) distance between
    endpoints.
    Reference implementation just for testing purpose.
    """
    e_A_1 = s_A[0]
    e_A_2 = s_A[-1]
    e_B_1 = s_B[0]
    e_B_2 = s_B[-1]
    d = norm(e_A_1 - e_B_1) + norm(e_A_2 - e_B_2)
    d_flip = norm(e_A_1 - e_B_2) + norm(e_A_2 - e_B_1)
    return min(d, d_flip)


def bundles_distances_endpoints(S_A, S_B):
    """Distance between lists/arrays or streamlines, based on
    endpoints. Returns the distance matrix between the related
    groups streamlines.
    Reference implementation just for testing purpose.
    """
    dm = np.empty((len(S_A), len(S_B)), dtype=np.float)
    for i, s_A in enumerate(S_A):
        for j, s_B in enumerate(S_B):
            dm[i, j] = streamline_distance_endpoints(s_A, s_B)

    return dm


def bundles_distances_endpoints_fast(S_A, S_B):
    """Distance between lists/arrays or streamlines, based on
    endpoints. Returns the distance matrix between the related groups
    streamlines.
    Fast implementation based on bundles_distances_mdf().
    """
    tmp_S_A = [[s_A[0], s_A[-1]] for s_A in S_A]
    tmp_S_B = [[s_B[0], s_B[-1]] for s_B in S_B]
    return 2.0 * bundles_distances_mdf(tmp_S_A, tmp_S_B)


def bundles_distances_endpoints_fastest(S_A, S_B):
    """Distance between lists/arrays or streamlines, based on
    endpoints. Returns the distance matrix between the related
    groups streamlines.
    Fastest implementation based on distance_matrix_mdf().
    """    
    tmp_S_A = np.array([[s_A[0], s_A[-1]] for s_A in S_A])
    tmp_S_B = np.array([[s_B[0], s_B[-1]] for s_B in S_B])
    return 2.0 * distance_matrix_mdf(tmp_S_A, tmp_S_B)


def compute_terminal_points_matrix(S_A, S_B):
    from dipy.tracking.streamline import set_number_of_points
    S_A_res = np.array([set_number_of_points(s, nb_points=2) for s in S_A])
    S_B_res = np.array([set_number_of_points(s, nb_points=2) for s in S_B])
    return 2.0 * bundles_distances_mdf(S_A_res, S_B_res)


if __name__ == '__main__':
    import numpy as np
    from time import time
    np.random.seed(0)

    s_A = np.random.uniform(size=[10, 3])
    s_B = np.random.uniform(size=[5, 3])
    print("Example of streamline_distance_endpoints: %s" % streamline_distance_endpoints(s_A, s_B))

    n_A = 1000
    n_B = 3000
    low = 5
    high = 200
    S_A = [np.random.uniform(size=[n, 3]) for n in np.random.randint(low, high, size=n_A)]
    S_B = [np.random.uniform(size=[n, 3]) for n in np.random.randint(low, high, size=n_B)]
    t0 = time()
    dm = bundles_distances_endpoints(S_A, S_B)
    print("bundles_distances_endpoints() : %s sec" % (time() - t0))
    t0 = time()
    dm_fast = bundles_distances_endpoints_fast(S_A, S_B)
    print("bundles_distances_endpoints_fast() : %s sec" % (time() - t0))
    print("max difference = %s" % np.abs(dm - dm_fast).max())
    t0 = time()
    dm_fastest = bundles_distances_endpoints_fastest(S_A, S_B)
    print("bundles_distances_endpoints_fastest() : %s sec" % (time() - t0))
    print("max difference = %s" % np.abs(dm - dm_fastest).max())
    t0 = time()
    dm_terminal = compute_terminal_points_matrix(S_A, S_B)
    print("compute_terminal_points_matrix() : %s sec" % (time() - t0))
    print("max difference = %s" % np.abs(dm - dm_terminal).max())

