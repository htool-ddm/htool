from __future__ import print_function
import sys
import ctypes
import numpy as np
import htool
import re
import matplotlib.pyplot as plt
import scipy.sparse.linalg as spla
from mpi4py import MPI
import math

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

scalar = ctypes.c_double

n = 1000

np.random.seed(0)
rho = np.random.random(n)
theta = np.random.random(n)
p=np.zeros((n,3))

p[:,0] = np.sqrt(rho)*np.cos(2*math.pi*theta)
p[:,1] = np.sqrt(rho)*np.sin(2*math.pi*theta)
p[:,2] = 1

@htool.getcoefFunc
def getcoef(i,j,r):
    r[0] = 1./(1e-5+math.sqrt(np.vdot(p[i]-p[j],p[i]-p[j])))

@htool.getsubmatrixFunc
def getsubmatrix(I,J,n,m,r):
    for i in range(0,n):
        for j in range(0,m):
            r[j*n+i] = 1./(1e-5+math.sqrt(np.vdot(p[I[i]]-p[J[j]],p[I[i]]-p[J[j]])))

#H = htool.HMatrixCreate(p, n, getcoef)
H = htool.HMatrixCreatewithsubmat(p, n, getsubmatrix)

def Hmv(v):
    res = np.zeros(n,dtype = scalar)
    htool.mvprod(H,v,res)
    return res

def iter(rk):
    if (rank == 0):
        print(np.linalg.norm(rk))

HOp = spla.LinearOperator((n, n), dtype = scalar, matvec=Hmv)

x = np.random.random(n)

[xh,infoh] = spla.gmres(HOp, x, callback = iter)

htool.printinfos(H)

htool.display(H)

A = np.zeros((n,n),dtype = scalar)

I = np.arange(n, dtype = np.int32)

for i in range(0,n):
    for j in range(0,n):
        A[i,j] = 1./(1e-5+np.vdot(p[i]-p[j],p[i]-p[j]));

def Amv(v):
    return A.dot(v)

AOp = spla.LinearOperator((n, n), dtype = scalar, matvec=Amv)

[xa,infoa] = spla.gmres(AOp, x, callback = iter)

if (rank == 0):
    print("err = ",np.linalg.norm(xh-xa)/np.linalg.norm(x))

#plt.scatter(p[:,0], p[:,1], c=xa, alpha=0.5)
#plt.gray()
#plt.show()