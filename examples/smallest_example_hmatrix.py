#!/usr/bin/env python
# coding: utf-8

import numpy as np
from numpy.linalg import norm
from scipy.sparse.linalg import gmres
from mpi4py import MPI

from htool.hmatrix import HMatrix

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# SETUP
# n² points on a regular grid in a square
n = int(np.sqrt(4761))
points = np.zeros((n*n, 3))
for j in range(0, n):
    for k in range(0, n):
        points[j+k*n, :] = (j, k, 1)

# BUILDING THE H-MATRIX
def get_coef(i, j, coef):
    coef[0] = 1.0 / (1e-5 + norm(points[i, :] - points[j, :]))

H = HMatrix.from_coefs(get_coef, points, symmetric=True, epsilon=1e-3, eta=100, minclustersize=10)

# def get_submatrix(I, J, n, m, coef):
#     for i in range(0,n):
#         for j in range(0,m):
#             coef[j*n+i] = 1.0 / (1.e-5 + norm(points[I[i], :] - points[J[j], :]))

# H = HMatrix.from_submatrices(points, get_submatrix, epsilon=1e-3, eta=100, minclustersize=10)

if rank == 0:
    print("Shape:", H.shape)
    print("Nb blocks:", H.nb_blocks)

H.print_infos()
H.display()
H.display_target_cluster(points,2)

full_H = 1.0 / (1e-5 + norm(points.reshape(1, H.shape[1], 3) - points.reshape(H.shape[0], 1, 3), axis=2))

# GMRES
y = np.ones((n*n,))
x, _ = gmres(H, y)
x_full, _ = gmres(full_H, y)

err_gmres_hmat  = norm(H @ x - y)
err_gmres_dense = norm(full_H @ x_full - y)
err_comp        = norm(x - x_full)/norm(x)

if rank==0:
    print("Error from gmres (Hmatrix):", err_gmres_hmat)
    print("Error from gmres (full matrix):", err_gmres_dense)
    print("Error between the two solutions:", err_comp)

