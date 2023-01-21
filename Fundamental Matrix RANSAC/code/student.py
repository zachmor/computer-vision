# Projection Matrix Stencil Code
# Written by Eleanor Tursman, based on previous work by Henry Hu, 
# Grady Williams, and James Hays for CSCI 1430 @ Brown and 
# CS 4495/6476 @ Georgia Tech

import numpy as np
import matplotlib.pyplot as plt
from skimage import io,color
from mpl_toolkits.mplot3d import Axes3D
import random

# Returns the projection matrix for a given set of corresponding 2D and
# 3D points. 
# 'Points_2D' is nx2 matrix of 2D coordinate of points on the image
# 'Points_3D' is nx3 matrix of 3D coordinate of points in the world
# 'M' is the 3x4 projection matrix
def calculate_projection_matrix(Points_2D, Points_3D):
    # To solve for the projection matrix. You need to set up a system of
    # equations using the corresponding 2D and 3D points:
    #
    #                                                     [M11       [ u1
    #                                                      M12         v1
    #                                                      M13         .
    #                                                      M14         .
    #[ X1 Y1 Z1 1 0  0  0  0 -u1*X1 -u1*Y1 -u1*Z1          M21         .
    #  0  0  0  0 X1 Y1 Z1 1 -v1*X1 -v1*Y1 -v1*Z1          M22         .
    #  .  .  .  . .  .  .  .    .     .      .          *  M23   =     .
    #  Xn Yn Zn 1 0  0  0  0 -un*Xn -un*Yn -un*Zn          M24         .
    #  0  0  0  0 Xn Yn Zn 1 -vn*Xn -vn*Yn -vn*Zn ]        M31         .
    #                                                      M32         un
    #                                                      M33         vn ]
    #
    # Then you can solve this using least squares with the '\' operator.
    # Notice you obtain 2 equations for each corresponding 2D and 3D point
    # pair. To solve this, you need at least 6 point pairs. Note that we set
    # M34 = 1 in this scenario. If you instead choose to use SVD, you should
    # not make this assumption and set up your matrices by following the 
    # set of equations on the project page. 
    #
    ##################
    # Your code here #
    ##################
    l = 2*len(Points_2D)

    A = np.ones((len(Points_3D), 4))
    A[:,:3] = Points_3D
    # print("A:", A)
    B = np.ones((len(Points_2D), 3))
    B[:,:2] = Points_2D
    # print("B:", B)
    A_test = np.zeros((l, 11))
    B_test = np.zeros((l))

    A_test[::2, :4] = A
    A_test[1::2, 4:8] = A
    A_test[::2, 8:] = Points_3D 
    A_test[1::2, 8:] = Points_3D 

    B_test = np.ravel(Points_2D)

    A_test[:, 8] *= -B_test
    A_test[:, 9] *= -B_test
    A_test[:, 10] *= -B_test
    # print("A-test:", A_test)
    # print("B-test:", B_test)
    x = np.append(np.linalg.lstsq(A_test, B_test, rcond=None)[0], 1)

    # print(np.linalg.lstsq(A_test, B_test)[0])
    # This M matrix came from a call to rand(3,4). It leads to a high residual.
    # Your total residual should be less than 1.
    # print('Randomly setting matrix entries as a placeholder')
    # M = np.array([[0.1768, 0.7018, 0.7948, 0.4613],
    #               [0.6750, 0.3152, 0.1136, 0.0480],
    #               [0.1020, 0.1725, 0.7244, 0.9932]]) 
    M = np.reshape(x, (3, 4))
    # M = np.linalg.lstsq(A, B)[0].T
    return M

# Returns the camera center matrix for a given projection matrix
# 'M' is the 3x4 projection matrix
# 'Center' is the 1x3 matrix of camera center location in world coordinates
def compute_camera_center(M):
    ##################
    # Your code here #
    ##################

    # Replace this with the correct code
    # In the visualization you will see that this camera location is clearly
    # incorrect, placing it in the center of the room where it would not see all
    # of the points.
    Q = -np.linalg.inv(M[:,:3])
    # print(Q)
    m4 = M[:,3]
    # print(m4)
    Center = np.matmul(Q,m4)

    return Center

# Returns the camera center matrix for a given projection matrix
# 'Points_a' is nx2 matrix of 2D coordinate of points on Image A
# 'Points_b' is nx2 matrix of 2D coordinate of points on Image B
# 'F_matrix' is 3x3 fundamental matrix
def estimate_fundamental_matrix(Points_a,Points_b):
    # Try to implement this function as efficiently as possible. It will be
    # called repeatly for part III of the project
    ##################
    # Your code here #
    #
    #                                                     [M11       [ u'1
    #                                                      M12         v'1
    #                                                      M13         .
    #                                                      M14         .
    #                                                      M21         .
    #      uu' vu' u' uv' vv' v' u v                      M22         .
    #        .  . .  .  .  .    .     .      .          *  M23   =     .
    #        Zn 1 0  0  0  0 -un*Xn -un*Yn -un*Zn          M24         .
    #        0  0 Xn Yn Zn 1 -vn*Xn -vn*Yn -vn*Zn ]        M31         .
    #                                                      M32         u'n
    #                                                      M33         v'n ]
    ##################
    l = len(Points_b)

    A = np.ones((len(Points_a), 9))
    
    A[:, :3] *= np.transpose([Points_b[:, 0] for i in range(3)])
    A[:, 3:6] *= np.transpose([Points_b[:, 1] for i in range(3)])
    A[:, ::3] *= np.transpose([Points_a[:, 0] for i in range(3)])
    A[:, 1::3] *= np.transpose([Points_a[:, 1] for i in range(3)])

    B_test = np.ravel(Points_b)

    # print("[!]",A_test)

    U, S, V = np.linalg.svd(A, full_matrices=True)
    F_matrix = V[-1].reshape(3,3)
    U, S, V = np.linalg.svd(F_matrix)
    S = np.diag(S)
    s_min_idx = np.argmin(S)
    S[s_min_idx] = 0
    # print("B:", B)
    # This is an intentionally incorrect Fundamental matrix placeholder
    # F_matrix = np.array([[0,0,-.0004],[0,0,.0032],[0,-0.0044,.1034]])
    # print("[!]", F_matrix)

    return F_matrix

# Find the best fundamental matrix using RANSAC on potentially matching
# points
# 'matches_a' and 'matches_b' are the Nx2 coordinates of the possibly
# matching points from pic_a and pic_b. Each row is a correspondence (e.g.
# row 42 of matches_a is a point that corresponds to row 42 of matches_b.
# 'Best_Fmatrix' is the 3x3 fundamental matrix
# 'inliers_a' and 'inliers_b' are the Mx2 corresponding points (some subset
# of 'matches_a' and 'matches_b') that are inliers with respect to
# Best_Fmatrix.
def ransac_fundamental_matrix(matches_a, matches_b):
    # For this section, use RANSAC to find the best fundamental matrix by
    # randomly sampling interest points. You would reuse
    # estimate_fundamental_matrix() from part 2 of this assignment.
    # If you are trying to produce an uncluttered visualization of epipolar
    # lines, you may want to return no more than 30 points for either left or
    # right images.
    ##################
    # Your code here #
    ##################

    # Your ransac loop should contain a call to 'estimate_fundamental_matrix()'
    # that you wrote for part II.
    threshold = .005

    def F_stats(indexes, Fmatrix):
        inliers_idx = []
        for i in indexes:
            x = np.append(matches_a[i], 1)
            x_prime = np.append(matches_b[i], 1)
            dist = np.matmul(np.matmul(x_prime, Fmatrix), x)
            # print("[!!!]", dist)
            if abs(dist) < threshold:
                inliers_idx.append(i)
        return inliers_idx

    Best_Fmatrix = estimate_fundamental_matrix(matches_a[0:15,:],matches_b[0:15,:])
    Best_inliers = list()
    index = np.arange(len(matches_a))
    counter = 0

    # for i in range(4000):
    #     Fmatrix = estimate_fundamental_matrix(matches_a[index[0:10],:],matches_b[index[0:10],:])
    #     inliers_idx = F_stats(index[0:10], Fmatrix)
    #     if len(inliers_idx) > len(Best_inliers):
    #         Best_inliers = inliers_idx
    #         Best_Fmatrix = Fmatrix
    #     random.shuffle(index)


    while len(Best_inliers) < 22:
        Fmatrix = estimate_fundamental_matrix(matches_a[index[0:9],:],matches_b[index[0:9],:])
        inliers_idx = F_stats(index[0:30], Fmatrix)
        if len(inliers_idx) > len(Best_inliers):
            Best_inliers = inliers_idx
            Best_Fmatrix = Fmatrix
        random.shuffle(index)
        counter += 1
    print("[!]: counter:", counter)
    # placeholders, you can delete all of this
    inliers_a = matches_a[Best_inliers]
    inliers_b = matches_b[Best_inliers]

    return Best_Fmatrix, inliers_a, inliers_b