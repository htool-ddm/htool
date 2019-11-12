#!/usr/bin/env python
# coding: utf-8

from time import perf_counter
from contextlib import contextmanager
import numpy as np
from numpy.linalg import norm
from scipy.sparse.linalg import gmres

from htool_test import HMatrix

@contextmanager
def timer(title=""):
    print("Start", title)
    start = perf_counter()
    try:
        yield None
    finally:
        end = perf_counter()
        print("Done! Elapsed time:", end - start)


# SETUP
# n² points on a regular grid in a square
n = int(np.sqrt(300))
points = np.zeros((n*n, 3))
for j in range(0, n):
    for k in range(0, n):
        points[j+k*n, :] = (j, k, 1)

def get_coef(i, j, coef):
    coef[0] = 1.0 / (1e-5 + norm(points[i, :] - points[j, :]))

def get_submatrix(I, J, n, m, coef):
    for i in range(0,n):
        for j in range(0,m):
            coef[j*n+i] = 1.0 / (1.e-5 + norm(points[I[i], :] - points[J[j], :]))

# BUILDING THE H-MATRIX
with timer("building HMatrix with coefs"):
    H = HMatrix.from_coefs(points, get_coef, epsilon=1e-3, eta=100, minclustersize=10)
with timer("building HMatrix with submatrix"):
    H = HMatrix.from_submatrices(points, get_submatrix, epsilon=1e-3, eta=100, minclustersize=10)

print("Shape:", H.shape)
print("Nb blocks:", H.nb_blocks)
# H.print_infos()
# H.display()

with timer("building full matrix"):
    full_H = 1.0 / (1e-5 + norm(points.reshape(1, n*n, 3) - points.reshape(n*n, 1, 3), axis=2))

# GMRES
y = np.ones((n*n,))

with timer("gmres on HMatrix"):
    x, _ = gmres(H, y)
print(norm(H @ x - y))

with timer("gmres on full matrix"):
    x_full, _ = gmres(full_H, y)
print(norm(full_H @ x_full - y))

print(norm(x - x_full)/norm(x))

