from __future__ import print_function
import sys
import ctypes
import numpy
import htool
import re
import matplotlib.pyplot as plt
import scipy.sparse.linalg as spla
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

scalar = ctypes.c_double

n = 1000

numpy.random.seed(0)
p = numpy.random.random((n,3))
p[:,2] = 0;

@htool.getcoefFunc
def getcoef(i,j,r):
    r._shape_ = (1,)
    r = numpy.ctypeslib.as_array(r)
    r[0] = 1./(1e-5+numpy.vdot(p[i]-p[j],p[i]-p[j]));

@htool.getsubmatrixFunc
def getsubmatrix(I,J,n,m,r):
    r._shape_ = (m,n)
    I._shape_ = (n,)
    J._shape_ = (m,)
    r = numpy.ctypeslib.as_array(r)
    I = numpy.ctypeslib.as_array(I)
    J = numpy.ctypeslib.as_array(J)
    for i in range(0,n):
        for j in range(0,m):
            r[j,i] = 1./(1e-5+numpy.vdot(p[I[i]]-p[J[j]],p[I[i]]-p[J[j]]));

#H = htool.HMatrixCreate(p, n, getcoef)
H = htool.HMatrixCreatewithsubmat(p, n, getsubmatrix)

def Hmv(v):
    res = numpy.zeros(n,dtype = scalar)
    htool.mvprod(H,v,res)
    return res

def iter(rk):
    if (rank == 0):
        print(numpy.linalg.norm(rk))

HOp = spla.LinearOperator((n, n), dtype = scalar, matvec=Hmv)

x = numpy.random.random(n)

[xh,infoh] = spla.gmres(HOp, x, callback = iter)

htool.printinfos(H)

A = numpy.zeros((n,n),dtype = scalar)

I = numpy.arange(n, dtype = numpy.int32)

getsubmatrix(I,I,n,n,A)

def Amv(v):
    return A.dot(v)

AOp = spla.LinearOperator((n, n), dtype = scalar, matvec=Amv)

[xa,infoa] = spla.gmres(AOp, x, callback = iter)

if (rank == 0):
    print("err = ",numpy.linalg.norm(xh-xa)/numpy.linalg.norm(x))

#plt.scatter(p[:,0], p[:,1], c=xa, alpha=0.5)
#plt.gray()
#plt.show()