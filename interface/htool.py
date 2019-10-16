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

getcoefFunc = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_int, ctypes.c_int)

class HMatrix(ctypes.Structure):
    pass
HMatrixCreate = lib.HMatrixCreate
HMatrixCreate.restype = ctypes.POINTER(HMatrix)
HMatrixCreate.argtypes = [ numpy.ctypeslib.ndpointer(dtype=numpy.float64,ndim=2,flags='C_CONTIGUOUS'), ctypes.c_int, getcoefFunc]

printinfos = lib.printinfos
printinfos.restype = None
printinfos.argtypes = [ ctypes.POINTER(HMatrix) ]

mvprod = lib.mvprod
mvprod.restype = None
mvprod.argtypes = [ ctypes.POINTER(HMatrix), numpy.ctypeslib.ndpointer(ctypes.c_double,flags='C_CONTIGUOUS'), numpy.ctypeslib.ndpointer(ctypes.c_double,flags='C_CONTIGUOUS')]
