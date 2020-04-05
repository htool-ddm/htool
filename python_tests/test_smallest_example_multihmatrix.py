#!/usr/bin/env python
# coding: utf-8

import numpy as np
from numpy.linalg import norm
from mpi4py import MPI

from htool.multihmatrix import MultiHMatrix


def smallest_example(m,n,submatrix):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # SETUP
    np.random.seed(0)
    points_target=np.zeros((m,3))
    points_target[:,0] = np.random.random(m)
    points_target[:,1] = np.random.random(m)
    points_target[:,2] = 1

    if n==m:
        points_source=points_target
    else:
        points_source=np.zeros((n,3))
        points_source[:,0] = np.random.random(n)
        points_source[:,1] = np.random.random(n)
        points_source[:,2] = 0
    print("ok")
    if not submatrix:
        # BUILDING THE H-MATRIX from get_coef
        def get_coefs(i, j, coefs):
            coefs = np.arange(1,2) / (1e-5 + norm(points_target[i, :] - points_source[j, :]))

        if n==m:
            print("ok")
            H = MultiHMatrix.from_coefs(get_coefs,5,points_target, epsilon=1e-6, eta=100, minclustersize=10)
        else:
            print("ok")
            H = MultiHMatrix.from_coefs(get_coefs,5,points_target,points_source, epsilon=1e-6, eta=100, minclustersize=10)

    # else:
        # # BUILDING THE H-MATRIX from get_submatrix
        # def get_submatrix(I, J, m, n, coef):
        #     for i in range(0,m):
        #         for j in range(0,n):
        #             coef[j*m+i] = 1.0 / (1.e-5 + norm(points_target[I[i], :] - points_source[J[j], :]))
        # if n==m:
        #     H = HMatrix.from_submatrices(get_submatrix,points_target, epsilon=1e-6, eta=100, minclustersize=10)
        # else:
        #     H = HMatrix.from_submatrices(get_submatrix,points_target,points_source, epsilon=1e-6, eta=100, minclustersize=10)
    
    # if rank == 0:
    #     print("Shape:", H.shape)
    #     print("Nb blocks:", H.nb_blocks)

    # H[0]

    # full_H = 1.0 / (1e-5 + norm(points_target.reshape(H.shape[0],1, 3) - points_source.reshape(1,H.shape[1], 3), axis=2))

    # x = np.random.rand(n)
    # y_1 = H.matvec(x)
    # y_2 = full_H.dot(x)
    # assert(norm(y_1-y_2)/norm(y_2)<1e-6)


def test_smallest_example():
    # smallest_example(1000,3000,True)
    smallest_example(100,300,False)
    # smallest_example(1000,1000,True)
    smallest_example(100,100,False)


test_smallest_example()