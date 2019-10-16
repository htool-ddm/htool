from mpi4py import MPI
import sys
import ctypes
import ctypes.util
_libc = ctypes.cdll.LoadLibrary(ctypes.util.find_library('c'))
import numpy
import re
if 'linux' in sys.platform:
    lib = ctypes.cdll.LoadLibrary('htool_python.so')
elif sys.platform == 'darwin':
    lib = ctypes.cdll.LoadLibrary('htool_python.dylib')
elif sys.platform == 'win32':
    lib = ctypes.cdll.LoadLibrary('htool_python.dll')

if ctypes.c_ushort.in_dll(lib, 'scalar').value == 0:
    scalar = ctypes.c_double
else:
    scalar = numpy.complex128

getcoefFunc = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_int, numpy.ctypeslib.ndpointer(scalar,flags='C_CONTIGUOUS'))

getsubmatrixFunc = ctypes.CFUNCTYPE(None, numpy.ctypeslib.ndpointer(ctypes.c_int,flags='C_CONTIGUOUS'), numpy.ctypeslib.ndpointer(ctypes.c_int,flags='C_CONTIGUOUS'), ctypes.c_int, ctypes.c_int, numpy.ctypeslib.ndpointer(scalar,flags='C_CONTIGUOUS'))

class HMatrix(ctypes.Structure):
    pass
HMatrixCreate = lib.HMatrixCreate
HMatrixCreate.restype = ctypes.POINTER(HMatrix)
HMatrixCreate.argtypes = [ numpy.ctypeslib.ndpointer(dtype=numpy.float64,ndim=2,flags='C_CONTIGUOUS'), ctypes.c_int, getcoefFunc]

HMatrixCreatewithsubmat = lib.HMatrixCreatewithsubmat
HMatrixCreatewithsubmat.restype = ctypes.POINTER(HMatrix)
HMatrixCreatewithsubmat.argtypes = [ numpy.ctypeslib.ndpointer(dtype=numpy.float64,ndim=2,flags='C_CONTIGUOUS'), ctypes.c_int, getsubmatrixFunc]

printinfos = lib.printinfos
printinfos.restype = None
printinfos.argtypes = [ ctypes.POINTER(HMatrix) ]

mvprod = lib.mvprod
mvprod.restype = None
mvprod.argtypes = [ ctypes.POINTER(HMatrix), numpy.ctypeslib.ndpointer(scalar,flags='C_CONTIGUOUS'), numpy.ctypeslib.ndpointer(scalar,flags='C_CONTIGUOUS')]
