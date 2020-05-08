import numpy as np
from htool.hmatrix import ComplexHMatrix
from scipy.sparse.linalg import gmres
from mpi4py import MPI
import math
import struct
import os

def python_gmres_get_coef():

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # Matrix
    with open(os.path.join(os.path.dirname(__file__)+"/../data/data_test/matrix.bin"), "rb" ) as input:
        data=input.read()
        (m, n) = struct.unpack("@II", data[:8])
        # print(m,n)
        A=np.frombuffer(data[8:],dtype=np.dtype('complex128'))
        A=np.transpose(A.reshape((m,n)))


    # Right-hand side
    with open(os.path.join(os.path.dirname(__file__)+"/../data/data_test/rhs.bin"), "rb" ) as input:
        data=input.read()
        l = struct.unpack("@I", data[:4])
        f=np.frombuffer(data[4:],dtype=np.dtype('complex128'))

    # mesh
    p=np.zeros((n,3))
    with open(os.path.join(os.path.dirname(__file__)+"/../data/data_test/mesh.msh"), "r" ) as input:
        check=False
        count=0
        for line in input:

            if line=="$EndNodes\n":
                break

            if check and len(line.split())==4:
                tab_line=line.split()
                p[count][0]=tab_line[1]
                p[count][1]=tab_line[2]
                p[count][2]=tab_line[3]
                count+=1

            if line=="$Nodes\n":
                check=True


    # Hmatrix
    def get_coef(i, j, coef):
        coef[0] = A[i][j].real
        coef[1] = A[i][j].imag

    H = ComplexHMatrix.from_coefs(get_coef, p , epsilon=1e-6, eta=0.1, minclustersize=1)

    # Global vectors
    with open(os.path.join(os.path.dirname(__file__)+"/../data/data_test/sol.bin"), "rb" ) as input:
        data=input.read()
        x_ref=np.frombuffer(data[4:],dtype=np.dtype('complex128'))

    # Solve
    x, _ = gmres(H, f,tol=1e-6)

    # Output
    H.print_infos()

    # Error on inversions
    error = np.linalg.norm(f-A.dot(x))/np.linalg.norm(f)

    if (rank==0):
        print("error: ",error)

    assert(error<1e-6)

def python_gmres_get_submatrix():

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # Matrix
    with open(os.path.join(os.path.dirname(__file__)+"/../data/data_test/matrix.bin"), "rb" ) as input:
        data=input.read()
        (m, n) = struct.unpack("@II", data[:8])
        A=np.frombuffer(data[8:],dtype=np.dtype('complex128'))
        A=np.transpose(A.reshape((m,n)))


    # Right-hand side
    with open(os.path.join(os.path.dirname(__file__)+"/../data/data_test/rhs.bin"), "rb" ) as input:
        data=input.read()
        l = struct.unpack("@I", data[:4])
        f=np.frombuffer(data[4:],dtype=np.dtype('complex128'))

    # mesh
    p=np.zeros((n,3))
    with open(os.path.join(os.path.dirname(__file__)+"/../data/data_test/mesh.msh"), "r" ) as input:
        check=False
        count=0
        for line in input:

            if line=="$EndNodes\n":
                break

            if check and len(line.split())==4:
                tab_line=line.split()
                p[count][0]=tab_line[1]
                p[count][1]=tab_line[2]
                p[count][2]=tab_line[3]
                count+=1

            if line=="$Nodes\n":
                check=True


    # Hmatrix
    def get_submatrix(I,J,n,m,r):
        for i in range(0,n):
            for j in range(0,m):
                r[2*(j*n+i)]   = A[I[i]][J[j]].real
                r[2*(j*n+i)+1] = A[I[i]][J[j]].imag

    H = ComplexHMatrix.from_submatrices(get_submatrix, p, epsilon=1e-6, eta=0.1, minclustersize=1)

    # Global vectors
    with open(os.path.join(os.path.dirname(__file__)+"/../data/data_test/sol.bin"), "rb" ) as input:
        data=input.read()
        x_ref=np.frombuffer(data[4:],dtype=np.dtype('complex128'))

    # Solve
    x, _ = gmres(H, f,tol=1e-6)

    # Output
    H.print_infos()

    # Error on inversions
    error = np.linalg.norm(f-A.dot(x))/np.linalg.norm(f)

    if (rank==0):
        print("error: ",error)

    assert(error<1e-6)



def test_python_gmres():
    python_gmres_get_coef()
    python_gmres_get_submatrix()