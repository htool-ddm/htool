#!/usr/bin/env python
# coding: utf-8

import os,sys
import ctypes
import numpy as np
from .hmatrix import _C_HMatrix, HMatrix


class _C_MultiHMatrix(ctypes.Structure):
    """Holder for the raw data from the C++ code."""
    pass


class AbstractMultiHMatrix:
    """Common code for the two actual MultiHMatrix classes below."""

    ndim = 2  # To mimic a numpy 2D array

    def __init__(self, c_data: _C_MultiHMatrix, **params):
        # Users should use one of the two constructors below.

        self.c_data = c_data
        self.shape = (self.lib.multi_nbrows(c_data), self.lib.multi_nbcols(c_data))
        self.size = self.lib.nbhmats(c_data)


        self.lib.getHMatrix.restype=ctypes.POINTER(_C_HMatrix)
        self.lib.getHMatrix.argtypes=[ctypes.POINTER(_C_MultiHMatrix), ctypes.c_int]

        self.hmatrices = []
        for l in range(0,self.size):
            c_data_hmatrix = self.lib.getHMatrix(self.c_data,l)
            self.hmatrices.append(HMatrix(c_data_hmatrix,**params))


        self.params = params.copy()

    @classmethod
    def from_coefs(cls, getcoefs, nm, points_target, points_source=None, **params):
        """Construct an instance of the class from a evaluation function.

        Parameters
        ----------
        getcoefs: Callable
            A function evaluating an array of matrices at given coordinates.
        points_target: np.ndarray of shape (N, 3)
            The coordinates of the target points. If points_source=None, also the coordinates of the target points
        points_source: np.ndarray of shape (N, 3)
            If not None; the coordinates of the source points.
        epsilon: float, keyword-only, optional
            Tolerance of the Adaptive Cross Approximation
        eta: float, keyword-only, optional
            Criterion to choose the blocks to compress
        minclustersize: int, keyword-only, optional
            Minimum shape of a block
        maxblocksize: int, keyword-only, optional
            Maximum number of coefficients in a block

        Returns
        -------
        MultiHMatrix or ComplexMultiHMatrix
        """
        # Set params.
        cls._set_building_params(**params)
        
        # Boilerplate code for Python/C++ interface.
        _getcoefs_func_type = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_double))
        if points_source is None:
            cls.lib.MultiHMatrixCreateSym.restype = ctypes.POINTER(_C_MultiHMatrix)
            cls.lib.MultiHMatrixCreateSym.argtypes = [
                np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
                ctypes.c_int,
                _getcoefs_func_type,
                ctypes.c_int
            ]

            # Call the C++ backend.
            c_data = cls.lib.MultiHMatrixCreateSym(points_target, points_target.shape[0], _getcoefs_func_type(getcoefs),nm)

        else:
            cls.lib.MultiHMatrixCreate.restype = ctypes.POINTER(_C_MultiHMatrix)
            cls.lib.MultiHMatrixCreate.argtypes = [
                np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
                ctypes.c_int,
                np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
                ctypes.c_int,
                _getcoefs_func_type,
                ctypes.c_int
            ]

            # Call the C++ backend.      
            c_data = cls.lib.MultiHMatrixCreate(points_target,points_target.shape[0],points_source, points_source.shape[0], _getcoefs_func_type(getcoefs),nm)

        return cls(c_data, **params)


    @classmethod
    def from_submatrices(cls, getsubmatrix, nm, points_target, points_source=None, **params):
        """Construct an instance of the class from a evaluation function.

        Parameters
        ----------
        points: np.ndarray of shape (N, 3)
            The coordinates of the points.
        getsubmatrix: Callable
            A function evaluating the matrix in a given range.
        epsilon: float, keyword-only, optional
            Tolerance of the Adaptive Cross Approximation
        eta: float, keyword-only, optional
            Criterion to choose the blocks to compress
        minclustersize: int, keyword-only, optional
            Minimum shape of a block
        maxblocksize: int, keyword-only, optional
            Maximum number of coefficients in a block

        Returns
        -------
        HMatrix or ComplexHMatrix
        """
        # Set params.
        cls._set_building_params(**params)

        # Boilerplate code for Python/C++ interface.
        _getsumatrix_func_type = ctypes.CFUNCTYPE(
                None, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
                ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_double)
            )
        if points_source is None:
            cls.lib.MultiHMatrixCreatewithsubmatSym.restype = ctypes.POINTER(_C_MultiHMatrix)
            cls.lib.MultiHMatrixCreatewithsubmatSym.argtypes = [
                np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
                ctypes.c_int,
                _getsumatrix_func_type,
                ctypes.c_int
            ]

            # Call the C++ backend.
            c_data = cls.lib.MultiHMatrixCreatewithsubmatSym(points_target, points_target.shape[0], _getsumatrix_func_type(getsubmatrix),nm)
        else:
            cls.lib.MultiHMatrixCreatewithsubmat.restype = ctypes.POINTER(_C_MultiHMatrix)
            cls.lib.MultiHMatrixCreatewithsubmat.argtypes = [
                np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
                ctypes.c_int,
                np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
                ctypes.c_int,
                _getsumatrix_func_type,
                ctypes.c_int
            ]

            # Call the C++ backend.
            c_data = cls.lib.MultiHMatrixCreatewithsubmat(points_target,points_target.shape[0],points_source, points_source.shape[0], _getsumatrix_func_type(getsubmatrix),nm)

        return cls(c_data, **params)

    @classmethod
    def _set_building_params(cls, *, eta=None, minclustersize=None, epsilon=None, maxblocksize=None):
        """Put the parameters in the C++ backend."""
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

    def __str__(self):
        return f"{self.__class__.__name__}(shape={self.shape})"

    def __getitem__(self, key):

        # self.lib.getHMatrix.restype=ctypes.POINTER(_C_HMatrix)
        # self.lib.getHMatrix.argtypes=[ctypes.POINTER(_C_MultiHMatrix), ctypes.c_int]
        # c_data_hmatrix = self.lib.getHMatrix(self.c_data,key)
        # return HMatrix(c_data_hmatrix,**self.params)
        return self.hmatrices[key]

    def matvec(self, l , vector):
        """Matrix-vector product (interface for scipy iterative solvers)."""

        assert self.shape[1] == vector.shape[0], "Matrix-vector product of matrices of wrong shapes."

        # Boilerplate for Python/C++ interface
        self.lib.MultiHMatrixVecProd.argtypes = [
                ctypes.POINTER(_C_MultiHMatrix),
                ctypes.c_int,
                np.ctypeslib.ndpointer(self.dtype, flags='C_CONTIGUOUS'),
                np.ctypeslib.ndpointer(self.dtype, flags='C_CONTIGUOUS')
                ]

        # Initialize vector
        result = np.zeros((self.shape[0],), dtype=self.dtype)

        # Call C++ backend
        self.lib.MultiHMatrixVecProd(self.c_data,l , vector, result)
        return result


class MultiHMatrix(AbstractMultiHMatrix):
    """A real-valued hierarchical matrix based on htool C++ library.
    Create with HMatrix.from_coefs or HMatrix.from_submatrices.

    Attributes
    ----------
    c_data:
        Pointer to the raw data used by the C++ library.
    shape: Tuple[int, int]
        Shape of the matrix.
    nb_dense_blocks: int
        Number of dense blocks in the hierarchical matrix.
    nb_low_rank_blocks: int
        Number of sparse blocks in the hierarchical matrix.
    nb_blocks: int
        Total number of blocks in the decomposition.
    params: dict
        The parameters that have been used to build the matrix.
    """
    libfile = os.path.join(os.path.dirname(__file__), '../libhtool_shared')
    if 'linux' in sys.platform:
        lib = ctypes.cdll.LoadLibrary(libfile+'.so')
    elif sys.platform == 'darwin':
        lib = ctypes.cdll.LoadLibrary(libfile+'.dylib')
    elif sys.platform == 'win32':
        lib = ctypes.cdll.LoadLibrary(libfile+'.dll')
    dtype = ctypes.c_double


class ComplexMultiHMatrix(AbstractMultiHMatrix):
    """A complex-valued hierarchical matrix based on htool C++ library.
    Create with ComplexHMatrix.from_coefs or ComplexHMatrix.from_submatrices.

    Attributes
    ----------
    c_data:
        Pointer to the raw data used by the C++ library.
    shape: Tuple[int, int]
        Shape of the matrix.
    nb_dense_blocks: int
        Number of dense blocks in the hierarchical matrix.
    nb_low_rank_blocks: int
        Number of sparse blocks in the hierarchical matrix.
    nb_blocks: int
        Total number of blocks in the decomposition.
    params: dict
        The parameters that have been used to build the matrix.
    """
    libfile = os.path.join(os.path.dirname(__file__), '../libhtool_shared_complex')
    if 'linux' in sys.platform:
        lib = ctypes.cdll.LoadLibrary(libfile+'.so')
    elif sys.platform == 'darwin':
        lib = ctypes.cdll.LoadLibrary(libfile+'.dylib')
    elif sys.platform == 'win32':
        lib = ctypes.cdll.LoadLibrary(libfile+'.dll')
    dtype = np.complex128

