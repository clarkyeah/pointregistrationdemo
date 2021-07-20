#!/usr/bin/python

import numpy as np

# Input: expects 3xN matrix of points
# Returns R,t
# R = 3x3 rotation matrix
# t = 3x1 column vector

def rigid_transform_3D(A, B):
    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # sanity check
    #if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2,:] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    return R, t


if __name__ == "__main__":
    # trial 1st
    # A =np.array([[1, 4.3368,-53.6819],[1, 4.7889,-56.6994],[1, 5.2542,-59.7153],[1, -3.9465,-64.4640],[1, -4.5972, -60.8229],[1, -5.2092, -57.1722]])
    # B =np.array([[1, 4.2622,-53.6927], [1, 4.7088,  -56.7115], [1, 5.1730,  -59.7280], [1, -3.9670,  -64.4679], [1, -4.6131,  -60.8256], [1, -5.2059, -57.1717]]) 

    # trial 2nd, switching order of trial 1st (1==>end)
    A =np.array([[1, 4.7889,-56.6994],[1, 5.2542,-59.7153],[1, -3.9465,-64.4640],[1, -4.5972, -60.8229],[1, -5.2092, -57.1722],[1, 4.3368,-53.6819]])
    B =np.array([[1, 4.7088,  -56.7115], [1, 5.1730,  -59.7280], [1, -3.9670,  -64.4679], [1, -4.6131,  -60.8256], [1, -5.2059, -57.1717],[1, 4.2622,-53.6927]]) 

    A = np.transpose(A)
    B = np.transpose(B)

    R, t = rigid_transform_3D(A, B)
    print(R)
    print(t)