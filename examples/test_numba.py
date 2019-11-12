#!/usr/bin/env python
# coding: utf-8

from contextlib import contextmanager
from numba import cfunc, types
from numpy.linalg import norm
from time import perf_counter
from mpi4py import MPI
import numpy as np

from htool_ import HMatrix

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

###############################################################################
# SETUP

@contextmanager
def timer(title="doing something"):
    if rank == 0:
        print("==> Start", title)
    start = perf_counter()
    try:
        yield None
    finally:
        end = perf_counter()
        if rank == 0:
            print("    Done! Elapsed time:", end - start)

params = dict(epsilon=1e-3, eta=1, minclustersize=10)

# n² points on a regular grid in a square
n = int(np.sqrt(5000))
points = np.zeros((n*n, 3))
for j in range(0, n):
    for k in range(0, n):
        points[j+k*n, :] = (j, k, 1)

###############################################################################
# PASSING A PYTHON GET_COEF

def get_coef(i, j, coef):
    coef[0] = 1.0 / (1e-5 + norm(points[i, :] - points[j, :]))

with timer("building HMatrix with Python get_coef"):
    H1 = HMatrix.from_coefs(points, get_coef, **params)

###############################################################################
# PASSING A PYTHON GET_SUBMATRIX

def get_submatrix(I, J, n, m, coef):
    for i in range(0,n):
        for j in range(0,m):
            coef[j*n+i] = 1.0 / (1e-5 + norm(points[I[i], :] - points[J[j], :]))

with timer("building HMatrix with Python get_submatrix"):
    H2 = HMatrix.from_submatrices(points, get_submatrix, **params)

###############################################################################
# PASSING A NUMBA COMPILED GET_COEF
# https://numba.pydata.org/numba-doc/dev/user/cfunc.html

@cfunc(types.void(types.intc, types.intc, types.CPointer(types.double)), nopython=True)
def get_coef_2(i, j, coef):
    coef[0] = 1.0 / (1e-5 + norm(points[i, :] - points[j, :]))

with timer("building HMatrix with Numba get_coef"):
    H3 = HMatrix.from_coefs(points, get_coef_2.ctypes, **params)

###############################################################################
# PASSING A NUMBA COMPILED GET_SUBMATRIX

@cfunc(types.void(types.CPointer(types.intc), types.CPointer(types.intc),
    types.intc, types.intc, types.CPointer(types.double)), nopython=True)
def get_submatrix_2(I, J, n, m, coef):
    for i in range(0, n):
        for j in range(0, m):
            coef[j*n+i] = 1.0 / (1e-5 + norm(points[I[i], :] - points[J[j], :]))

with timer("building HMatrix with Numba get_submatrix"):
    H4 = HMatrix.from_submatrices(points, get_submatrix_2.ctypes, **params)

###############################################################################
# FULL MATRIX

with timer("building full matrix"):
    full_H = 1.0 / (1e-5 + norm(points.reshape(1, n*n, 3) - points.reshape(n*n, 1, 3), axis=2))

