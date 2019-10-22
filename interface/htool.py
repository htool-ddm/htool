from mpi4py import MPI
import sys
import ctypes
import ctypes.util
_libc = ctypes.cdll.LoadLibrary(ctypes.util.find_library('c'))
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as patches

if 'linux' in sys.platform:
    lib = ctypes.cdll.LoadLibrary('htool_python.so')
elif sys.platform == 'darwin':
    lib = ctypes.cdll.LoadLibrary('htool_python.dylib')
elif sys.platform == 'win32':
    lib = ctypes.cdll.LoadLibrary('htool_python.dll')

if ctypes.c_ushort.in_dll(lib, 'scalar').value == 0:
    scalar = ctypes.c_double
else:
    scalar = np.complex128

getcoefFunc = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_int, ctypes.POINTER(scalar))

getsubmatrixFunc = ctypes.CFUNCTYPE(None, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int, ctypes.POINTER(scalar))

class HMatrix(ctypes.Structure):
    pass
HMatrixCreate = lib.HMatrixCreate
HMatrixCreate.restype = ctypes.POINTER(HMatrix)
HMatrixCreate.argtypes = [ np.ctypeslib.ndpointer(dtype=np.float64,ndim=2,flags='C_CONTIGUOUS'), ctypes.c_int, getcoefFunc]

HMatrixCreatewithsubmat = lib.HMatrixCreatewithsubmat
HMatrixCreatewithsubmat.restype = ctypes.POINTER(HMatrix)
HMatrixCreatewithsubmat.argtypes = [ np.ctypeslib.ndpointer(dtype=np.float64,ndim=2,flags='C_CONTIGUOUS'), ctypes.c_int, getsubmatrixFunc]

nbrows = lib.nbrows
nbrows.restype = ctypes.c_int
nbrows.argtypes = [ ctypes.POINTER(HMatrix) ]

nbcols = lib.nbcols
nbcols.restype = ctypes.c_int
nbcols.argtypes = [ ctypes.POINTER(HMatrix) ]

getndmat = lib.getndmat
getndmat.restype = ctypes.c_int
getndmat.argtypes = [ ctypes.POINTER(HMatrix) ]

getnlrmat = lib.getnlrmat
getnlrmat.restype = ctypes.c_int
getnlrmat.argtypes = [ ctypes.POINTER(HMatrix) ]

getpattern = lib.getpattern
getpattern.restype = None
getpattern.argtypes = [ ctypes.POINTER(HMatrix), np.ctypeslib.ndpointer(ctypes.c_int,flags='C_CONTIGUOUS')]

printinfos = lib.printinfos
printinfos.restype = None
printinfos.argtypes = [ ctypes.POINTER(HMatrix) ]

mvprod = lib.mvprod
mvprod.restype = None
mvprod.argtypes = [ ctypes.POINTER(HMatrix), np.ctypeslib.ndpointer(scalar,flags='C_CONTIGUOUS'), np.ctypeslib.ndpointer(scalar,flags='C_CONTIGUOUS')]

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def display(H):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    nbd = getndmat(H)
    nblr = getnlrmat(H)
    nb = nbd+nblr
    buf = np.zeros(5*nb,dtype = ctypes.c_int)
    getpattern(H,buf)
    if (rank == 0):
        nr = nbrows(H)
        nc = nbcols(H)
        matrix = np.zeros((nr,nc))
        fig, axes = plt.subplots(1,1)
        axes.xaxis.tick_top()
        plt.imshow(matrix)

        # Issue: there a shift of one pixel along the y-axis...
        shift = axes.transData.transform([(0,0), (1,1)])
        shift = shift[1,1] - shift[0,1]  # 1 unit in display coords
        shift = 1/shift  # 1 pixel in display coords

        # Loop
        for i in range(0,nb):
            matrix[np.ix_(range(buf[5*i],buf[5*i]+buf[5*i+1]),range(buf[5*i+2],buf[5*i+2]+buf[5*i+3]))]=buf[5*i+4]
            rect = patches.Rectangle((buf[5*i+2]-0.5,buf[5*i]-0.5+shift),buf[5*i+3],buf[5*i+1],linewidth=0.75,edgecolor='k',facecolor='none')
            axes.add_patch(rect)
            if buf[5*i+4]>=0 and buf[5*i+3]/float(nc)>0.05 and buf[5*i+1]/float(nc)>0.05:
                axes.annotate(buf[5*i+4],(buf[5*i+2]+buf[5*i+3]/2.,buf[5*i]+buf[5*i+1]/2.),color="white",size=10, va='center', ha='center')

        cmap = plt.get_cmap('YlGn')
        new_cmap = truncate_colormap(cmap, 0.4, 1)

        # Plot
        matrix =np.ma.masked_where(0>matrix,matrix)
        new_cmap.set_bad(color="red")
        plt.imshow(matrix,cmap=new_cmap,vmin=0, vmax=10)
        plt.show()
