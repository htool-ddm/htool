#!/usr/bin/env python
# coding: utf-8

import os
import ctypes
import numpy as np
from mpi4py import MPI

 
class C_HMatrix(ctypes.Structure):
    pass
 

class AbstractHMatrix:

    ndim = 2

    def __init__(self, c_data: C_HMatrix, **params):
        self.c_data = c_data

        self.shape = (self.lib.nbrows(c_data), self.lib.nbcols(c_data))

        self.nb_dense_blocks = self.lib.getndmat(c_data)
        self.nb_low_rank_blocks = self.lib.getnlrmat(c_data)
        self.nb_blocks = self.nb_dense_blocks + self.nb_low_rank_blocks

        self.params = params

    @classmethod
    def _set_building_params(cls, *, eta=None, minclustersize=None, epsilon=None, maxblocksize=None):
        if epsilon is not None:
            cls.lib.setepsilon.restype = None
            cls.lib.setepsilon.argtypes = [ ctypes.c_double ]
            cls.lib.setepsilon(epsilon)

        if eta is not None:
            cls.lib.seteta.restype = None
            cls.lib.seteta.argtypes = [ ctypes.c_double ]
            cls.lib.seteta(eta)

        if minclustersize is not None:
            cls.lib.setminclustersize.restype = None
            cls.lib.setminclustersize.argtypes = [ ctypes.c_int ]
            cls.lib.setminclustersize(minclustersize)
         
        if maxblocksize is not None:
            cls.lib.setmaxblocksize.restype = None
            cls.lib.setmaxblocksize.argtypes = [ ctypes.c_int ]
            cls.lib.setmaxblocksize(maxblocksize)

    @classmethod
    def from_coefs(cls, points, getcoef, **params):
        cls._set_building_params(**params)
        _getcoef_func_type = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_int, ctypes.POINTER(cls.dtype))
        cls.lib.HMatrixCreate.restype = ctypes.POINTER(C_HMatrix)
        cls.lib.HMatrixCreate.argtypes = [np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'), ctypes.c_int, _getcoef_func_type]
        c_data = cls.lib.HMatrixCreate(points, points.shape[0], _getcoef_func_type(getcoef))
        return cls(c_data, **params)

    @classmethod
    def from_submatrices(cls, points, getsubmatrix, **params):
        cls._set_building_params(**params)
        _getsumatrix_func_type = ctypes.CFUNCTYPE(None, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int, ctypes.POINTER(cls.dtype))
        cls.lib.HMatrixCreatewithsubmat.restype = ctypes.POINTER(C_HMatrix)
        cls.lib.HMatrixCreatewithsubmat.argtypes = [np.ctypeslib.ndpointer(dtype=cls.dtype, ndim=2, flags='C_CONTIGUOUS'), ctypes.c_int, _getsumatrix_func_type]
        c_data = cls.lib.HMatrixCreatewithsubmat(points, points.shape[0], _getsumatrix_func_type(getsubmatrix))
        return cls(c_data, **params)

    def __str__(self):
        return f"{self.__class__.__name__}(shape={self.shape}, nb_dense_blocks={self.nb_dense_blocks}, nb_low_rank_blocks={self.nb_low_rank_blocks})"

    def matvec(self, other):
        assert self.shape[1] == other.shape[0]
        result = np.zeros((self.shape[0],), dtype=self.dtype)
        self.lib.mvprod.argtypes = [
                ctypes.POINTER(C_HMatrix),
                np.ctypeslib.ndpointer(self.dtype, flags='C_CONTIGUOUS'),
                np.ctypeslib.ndpointer(self.dtype, flags='C_CONTIGUOUS')
                ]
        self.lib.mvprod(self.c_data, other, result)
        return result

    def __matmul__(self, other):
        if isinstance(other, np.ndarray) and len(other.shape) == 1:
            # other is a vector
            return self.matvec(other)
        else:
            return NotImplemented

    def print_infos(self):
        self.lib.printinfos(self.c_data)

    def _pattern(self):
        buf = np.zeros(5*self.nb_blocks, dtype=ctypes.c_int)

        self.lib.getpattern.restype = None
        self.lib.getpattern.argtypes = [ ctypes.POINTER(C_HMatrix), np.ctypeslib.ndpointer(ctypes.c_int, flags='C_CONTIGUOUS')]
        self.lib.getpattern(self.c_data, buf)

        return buf

    def display(self):
        if (MPI.COMM_WORLD.Get_rank() == 0):
            import matplotlib.pyplot as plt
            from matplotlib.patches import Rectangle
            cmap = plt.get_cmap('YlGn')
            max_rank = 10

            buf = self._pattern()

            plt.figure()
            ax = plt.gca()

            for i in range(0, self.nb_blocks):
                i_row, nb_row, i_col, nb_col, rank = buf[5*i:5*i+5]

                if rank < 0:
                    color = 'red'
                else:
                    color = cmap(rank/max_rank)

                rect = Rectangle(
                    (i_col-0.5, i_row-0.5), nb_col, nb_row,
                    linewidth=0.75, edgecolor='k', facecolor=color,
                )
                ax.add_patch(rect)

                if rank >= 0 and nb_row > 0.05*self.shape[0] and nb_col > 0.05*self.shape[1]:
                    ax.annotate(
                        str(rank), (i_col + nb_col/2, i_row + nb_row/2),
                        color="white", size=10, va='center', ha='center',
                    )

            plt.axis('equal')
            ax.set(xlim=(0, self.shape[0]), ylim=(0, self.shape[1]))
            ax.xaxis.tick_top()
            ax.invert_yaxis()
            plt.show()



class HMatrix(AbstractHMatrix):
    libfile = os.path.join(os.path.dirname(__file__), 'libhtool_shared')
    lib = ctypes.cdll.LoadLibrary(libfile + '.so')
    dtype = ctypes.c_double


class ComplexHMatrix(AbstractHMatrix):
    libfile = os.path.join(os.path.dirname(__file__), 'libhtool_shared_complex')
    lib = ctypes.cdll.LoadLibrary(libfile + '.so')
    dtype = np.complex128

