from __future__ import print_function
import sys
import numpy as np
import htool_complex
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
htool_complex.set_epsilon(1e-6)
htool_complex.set_eta(0.1)
htool_complex.set_minclustersize(1)


# Matrix
with open("data/data_test/SPD_mat_example.bin", "rb" ) as input:
    data=input.read()
    (m, n) = struct.unpack("@II", data[:8])
    # print(m,n)
    A=np.frombuffer(data[8:],dtype=np.dtype('complex128'))
    A=np.transpose(A.reshape((m,n)))


# Right-hand side
with open("data/data_test/rhs.bin", "rb" ) as input:
    data=input.read()
    l = struct.unpack("@I", data[:4])
    f=np.frombuffer(data[4:],dtype=np.dtype('complex128'))

# mesh
gmsh.initialize(sys.argv)
gmsh.merge("data/data_test/gmsh_mesh.msh")
p=gmsh.model.mesh.getNodes()[1]
p=p.reshape((n,3))

# Hmatrix
    
@htool_complex.getcoefFunc
def getcoef(i,j,r):
    r[0] = A[i][j].real
    r[1] = A[i][j].imag


@htool_complex.getsubmatrixFunc
def getsubmatrix(I,J,n,m,r):
    for i in range(0,n):
        for j in range(0,m):
            r[2*(j*n+i)]   = A[i][j].real
            r[2*(j*n+i)+1] = A[i][j].imag


H = htool_complex.HMatrixCreate(p, n, getcoef)
# H = htool_complex.HMatrixCreatewithsubmat(p, n, getsubmatrix)

# Global vectors
with open("data/data_test/sol.bin", "rb" ) as input:
    data=input.read()
    x_ref=np.frombuffer(data[4:],dtype=np.dtype('complex128'))

# Solve
def Hmv(v):
    res = np.zeros(n,dtype = scalar)
    htool_complex.mvprod(H,v,res)
    return res

def iter(rk):
    if (rank == 0):
        print(np.linalg.norm(rk))


HOp = spla.LinearOperator((n, n), dtype = scalar, matvec=Hmv)

[x,info] = spla.gmres(HOp, f, callback = iter,tol=1e-06)

# Output
htool_complex.printinfos(H)
# htool_complex.display(H)

# Error on inversions
inv_error = np.linalg.norm(f-A.dot(x))/np.linalg.norm(f)
error     = np.linalg.norm(x-x_ref)/np.linalg.norm(x_ref)
assert(inv_error<1e-6)
assert(error<1e-6)

if (rank==0):
    print("error on inversion : ",inv_error)
    print("error on solution : ",error)
