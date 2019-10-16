from __future__ import print_function
import sys
import ctypes
import numpy
import htool
import re
import matplotlib.pyplot as plt
import scipy.sparse.linalg as spla

n = 1000

p = numpy.random.random((n,3))
p[:,2] = 0;

@htool.getcoefFunc
def getcoef(i,j):
    return 1./(1e-5+numpy.vdot(p[i]-p[j],p[i]-p[j]));

H = htool.HMatrixCreate(p, n, getcoef)

htool.printinfos(H)

x = numpy.random.random(n)
x[:] = 1

y = numpy.random.random(n)

def mv(v):
    res = numpy.zeros(n)
    htool.mvprod(H,v,res)
    return res

def iter(rk):
    print(numpy.linalg.norm(rk))

A = spla.LinearOperator((n, n), matvec=mv)

spla.gmres(A, x, callback = iter)

plt.scatter(p[:,0], p[:,1], c=y, alpha=0.5)
plt.gray()
plt.show()