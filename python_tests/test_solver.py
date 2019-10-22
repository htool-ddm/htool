from __future__ import print_function
import sys
import ctypes
import numpy as np
import htool_complex
import re
import matplotlib.pyplot as plt
import scipy.sparse.linalg as spla
from mpi4py import MPI
import math
import gmsh_api.gmsh as gmsh
import struct

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

scalar = np.complex128


# Htool parameters
htool_complex.set_epsilon(1e-3)
htool_complex.set_eta(100)
htool_complex.set_minclustersize(10)


# Matrix
with open("data/data_test/SPD_mat_example.bin", "rb" ) as input:
    data=input.read()
    (m, n) = struct.unpack("@II", data[:8])
    # print(m,n)
    A=np.frombuffer(data[8:],dtype=np.dtype('complex128'))
    A=A.reshape((m,n))
    # print(A,A.shape)


# Right-hand side
with open("data/data_test/rhs.bin", "rb" ) as input:
    data=input.read()
    l = struct.unpack("@I", data[:4])
    print(l)
    f=np.frombuffer(data[4:],dtype=np.dtype('complex128'))
    # print(f)

# mesh
gmsh.initialize(sys.argv)
gmsh.merge("data/data_test/gmsh_mesh.msh")
p=gmsh.model.mesh.getNodes()[1]
p=p.reshape((n,3))

# Hmatrix
    
@htool_complex.getcoefFunc
def getcoef(i,j,r):
    r[0] = A[i][j]


@htool_complex.getsubmatrixFunc
def getsubmatrix(I,J,n,m,r):
    for i in range(0,n):
        for j in range(0,m):
            r[j*n+i] = A[i][j]


H = htool_complex.HMatrixCreate(p, n, getcoef)
# H = htool_complex.HMatrixCreatewithsubmat(p, n, getsubmatrix)

# Global vectors
x = np.zeros(n)
with open("data/data_test/rhs.bin", "rb" ) as input:
    data=input.read()
    x_ref=np.frombuffer(data[4:],dtype=np.dtype('complex128'))
    # print(x_ref)

# Solve
def Hmv(v):
    res = np.zeros(n,dtype = scalar)
    htool_complex.mvprod(H,v,res)
    return res

def iter(rk):
    if (rank == 0):
        print(np.linalg.norm(rk))


HOp = spla.LinearOperator((n, n), dtype = scalar, matvec=Hmv)

[xh_sol,infoh] = spla.gmres(HOp, x, callback = iter)

# Output
htool_complex.printinfos(H)
htool_complex.display(H)

# Error on inversions
inv_error = np.linalg.norm(f-A.dot(xh_sol))/np.linalg.norm(f)
error     = np.linalg.norm(xh_sol-x_ref)/np.linalg.norm(x_ref)

if (rank==0):
    print("error on inversion : ",inv_error)
    print("error on solution : ",error)
