#!/usr/bin/env python
# coding: utf-8

import os,sys
import ctypes
import numpy as np
from mpi4py import MPI


class _C_HMatrix(ctypes.Structure):
    """Holder for the raw data from the C++ code."""
    pass


class AbstractHMatrix:
    """Common code for the two actual HMatrix classes below."""

    ndim = 2  # To mimic a numpy 2D array

    def __init__(self, c_data: _C_HMatrix, **params):
        # Users should use one of the two constructors below.
        self.c_data = c_data
        self.shape = (self.lib.nbrows(c_data), self.lib.nbcols(c_data))
        self.nb_dense_blocks = self.lib.getndmat(c_data)
        self.nb_low_rank_blocks = self.lib.getnlrmat(c_data)
        self.nb_blocks = self.nb_dense_blocks + self.nb_low_rank_blocks
        self.params = params.copy()

    @classmethod
    def from_coefs(cls, getcoef, points_target, points_source=None, symmetric=False, **params):
        """Construct an instance of the class from a evaluation function.

        Parameters
        ----------
        getcoef: Callable
            A function evaluating the matrix at given coordinates.
        points_target: np.ndarray of shape (N, 3)
            The coordinates of the target points. If points_source=None, also the coordinates of the target points
        points_source: np.ndarray of shape (N, 3)
            If not None; the coordinates of the source points.
        symmetric: boolean
            If True, builds a symmetric HMatrix. Default to False.
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
        _getcoef_func_type = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_double))
        if points_source is None:
            cls.lib.HMatrixCreateSym.restype = ctypes.POINTER(_C_HMatrix)
            cls.lib.HMatrixCreateSym.argtypes = [
                np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
                ctypes.c_int,
                _getcoef_func_type,
                ctypes.c_bool
            ]

            # Call the C++ backend.
            c_data = cls.lib.HMatrixCreateSym(points_target, points_target.shape[0], _getcoef_func_type(getcoef),symmetric)

        else:
            cls.lib.HMatrixCreate.restype = ctypes.POINTER(_C_HMatrix)
            cls.lib.HMatrixCreate.argtypes = [
                np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
                ctypes.c_int,
                np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
                ctypes.c_int,
                _getcoef_func_type,
            ]

            # Call the C++ backend.            
            c_data = cls.lib.HMatrixCreate(points_target,points_target.shape[0],points_source, points_source.shape[0], _getcoef_func_type(getcoef))

        return cls(c_data, **params)


    @classmethod
    def from_submatrices(cls, getsubmatrix, points_target, points_source=None, symmetric=False, **params):
        """Construct an instance of the class from a evaluation function.

        Parameters
        ----------
        points: np.ndarray of shape (N, 3)
            The coordinates of the points.
        getsubmatrix: Callable
            A function evaluating the matrix in a given range.
        symmetric: boolean
        If True, builds a symmetric HMatrix. Default to False.
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
                ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_double),
            )
        if points_source is None:
            cls.lib.HMatrixCreatewithsubmatSym.restype = ctypes.POINTER(_C_HMatrix)
            cls.lib.HMatrixCreatewithsubmatSym.argtypes = [
                np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
                ctypes.c_int,
                _getsumatrix_func_type,
                ctypes.c_bool
            ]

            # Call the C++ backend.
            c_data = cls.lib.HMatrixCreatewithsubmatSym(points_target, points_target.shape[0], _getsumatrix_func_type(getsubmatrix),symmetric)
        else:
            cls.lib.HMatrixCreatewithsubmat.restype = ctypes.POINTER(_C_HMatrix)
            cls.lib.HMatrixCreatewithsubmat.argtypes = [
                np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
                ctypes.c_int,
                np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
                ctypes.c_int,
                _getsumatrix_func_type,
            ]

            # Call the C++ backend.
            c_data = cls.lib.HMatrixCreatewithsubmat(points_target,points_target.shape[0],points_source, points_source.shape[0], _getsumatrix_func_type(getsubmatrix))

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
        return f"{self.__class__.__name__}(shape={self.shape}, nb_dense_blocks={self.nb_dense_blocks}, nb_low_rank_blocks={self.nb_low_rank_blocks})"

    def matvec(self, vector):
        """Matrix-vector product (interface for scipy iterative solvers)."""

        assert self.shape[1] == vector.shape[0], "Matrix-vector product of matrices of wrong shapes."

        # Boilerplate for Python/C++ interface
        self.lib.mvprod.argtypes = [
                ctypes.POINTER(_C_HMatrix),
                np.ctypeslib.ndpointer(self.dtype, flags='C_CONTIGUOUS'),
                np.ctypeslib.ndpointer(self.dtype, flags='C_CONTIGUOUS')
                ]

        # Initialize vector
        result = np.zeros((self.shape[0],), dtype=self.dtype)

        # Call C++ backend
        self.lib.mvprod(self.c_data, vector, result)

        return result

    def __matmul__(self, other):
        if isinstance(other, np.ndarray) and len(other.shape) == 1:  # other is a vector
            return self.matvec(other)
        else:
            return NotImplemented

    def print_infos(self):
        self.lib.printinfos(self.c_data)

    def _pattern(self):
        buf = np.zeros(5*self.nb_blocks, dtype=ctypes.c_int)

        self.lib.getpattern.restype = None
        self.lib.getpattern.argtypes = [ ctypes.POINTER(_C_HMatrix), np.ctypeslib.ndpointer(ctypes.c_int, flags='C_CONTIGUOUS')]
        self.lib.getpattern(self.c_data, buf)

        return buf

    def _target_cluster(self,points_target,depth):
        output = np.zeros(4*len(points_target), dtype=ctypes.c_double)

        self.lib.get_target_cluster.restype = None
        self.lib.get_target_cluster.argtypes = [ ctypes.POINTER(_C_HMatrix), np.ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),np.ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),ctypes.c_int]
        self.lib.get_target_cluster(self.c_data,points_target,output,depth)

        return output

    def _source_cluster(self,points_source,depth):
        output = np.zeros(4*len(points_source), dtype=ctypes.c_double)

        self.lib.get_source_cluster.restype = None
        self.lib.get_source_cluster.argtypes = [ ctypes.POINTER(_C_HMatrix), np.ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),np.ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),ctypes.c_int]
        self.lib.get_source_cluster(self.c_data,points_source,output,depth)

        return output

    def display(self):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        import matplotlib.patches as patches
        import matplotlib.colors as colors

        buf = self._pattern()

        if MPI.COMM_WORLD.Get_rank() == 0:

            # First Data
            nr = self.shape[0]
            nc = self.shape[1]
            matrix = np.zeros((nr,nc))

            # Figure
            fig, axes = plt.subplots(1,1)
            axes.xaxis.tick_top()
            plt.imshow(matrix)

            # Issue: there a shift of one pixel along the y-axis...
            shift = axes.transData.transform([(0,0), (1,1)])
            shift = shift[1,1] - shift[0,1]  # 1 unit in display coords
            shift = 0


            for i in range(0, self.nb_blocks):
                i_row, nb_row, i_col, nb_col, rank = buf[5*i:5*i+5]

                matrix[np.ix_(range(i_row,i_row+nb_row),range(i_col,i_col+nb_col))]=rank

                rect = patches.Rectangle((i_col-0.5,i_row-0.5+shift),nb_col,nb_row,linewidth=0.75,edgecolor='k',facecolor='none')
                axes.add_patch(rect)
                
                if rank>=0 and nb_col/float(nc)>0.05 and nb_row/float(nc)>0.05:
                    axes.annotate(rank,(i_col+nb_col/2.,i_row+nb_row/2.),color="white",size=10, va='center', ha='center')

            # Colormap
            def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
                new_cmap = colors.LinearSegmentedColormap.from_list(
                    'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
                    cmap(np.linspace(minval, maxval, n)))
                return new_cmap

            cmap = plt.get_cmap('YlGn')
            new_cmap = truncate_colormap(cmap, 0.4, 1)

            # Plot
            matrix =np.ma.masked_where(0>matrix,matrix)
            new_cmap.set_bad(color="red")
            plt.imshow(matrix,cmap=new_cmap,vmin=0, vmax=10)

            plt.show()


    def display_target_cluster(self,points_target,depth):
        from mpl_toolkits.mplot3d import Axes3D 
        import matplotlib.pyplot as plt
        import matplotlib.colors as colors

        if MPI.COMM_WORLD.Get_rank() == 0:
            size = len(points_target)
            output = self._target_cluster(points_target,depth)
            print(output,size)
            print(output[3*size:])
            # Create Color Map
            colormap = plt.get_cmap("Dark2")
            norm = colors.Normalize(vmin=min(output[3*size:]), vmax=max(output[3*size:]))

            # Figure
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            ax.scatter(output[0:size], output[size:2*size], output[2*size:3*size],c=colormap(norm(output[3*size:])), marker='o')

            plt.show()

    def display_source_cluster(self,points_target,depth):
        from mpl_toolkits.mplot3d import Axes3D 
        import matplotlib.pyplot as plt
        import matplotlib.colors as colors

        if MPI.COMM_WORLD.Get_rank() == 0:
            size = len(points_target)
            output = self._target_cluster(points_target,depth)
            print(output,size)
            print(output[3*size:])
            # Create Color Map
            colormap = plt.get_cmap("Dark2")
            norm = colors.Normalize(vmin=min(output[3*size:]), vmax=max(output[3*size:]))

            # Figure
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            ax.scatter(output[0:size], output[size:2*size], output[2*size:3*size],c=colormap(norm(output[3*size:])), marker='o')

            plt.show()
            
            

class HMatrix(AbstractHMatrix):
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


class ComplexHMatrix(AbstractHMatrix):
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

