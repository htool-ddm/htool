#!/usr/bin/env python
# coding: utf-8

import numpy as np
from numpy.linalg import norm
from scipy.sparse.linalg import gmres
from mpi4py import MPI

from htool.multihmatrix import MultiHMatrix

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nm=2

# SETUP
# n² points on a regular grid in a square
n = int(np.sqrt(4761))
points = np.zeros((n*n, 3))
for j in range(0, n):
    for k in range(0, n):
        points[j+k*n, :] = (j, k, 1)

# BUILDING THE H-MATRIX
def get_coefs(i, j, coefs):
    for l in range(0,nm):
        coefs[l]=(l+1) / (1e-5+ norm(points[i, :] - points[j, :]))


H = MultiHMatrix.from_coefs(get_coefs,nm,points, epsilon=1e-3, eta=100, minclustersize=10)

for l in range(0,nm):
    if rank == 0:
        print("Shape:", H[l].shape)
        print("Nb blocks:", H[l].nb_blocks)

    H[l].print_infos()
    H[l].display()
    H[l].display_target_cluster(points,2)

    full_H = (l+1) / (1e-5 + norm(points.reshape(1, H.shape[1], 3) - points.reshape(H.shape[0], 1, 3), axis=2))

    # GMRES
    y = np.ones((n*n,))
    x, _ = gmres(H[l], y)
    x_full, _ = gmres(full_H, y)

    err_gmres_hmat  = norm(H[l] @ x - y)
    err_gmres_dense = norm(full_H @ x_full - y)
    err_comp        = norm(x - x_full)/norm(x)

    if rank==0:
        print("Error from gmres (Hmatrix):", err_gmres_hmat)
        print("Error from gmres (full matrix):", err_gmres_dense)
        print("Error between the two solutions:", err_comp)

