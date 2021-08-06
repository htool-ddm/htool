# HTOOL [![CI](https://github.com/htool-ddm/htool/actions/workflows/CI.yml/badge.svg?branch=develop)](https://github.com/htool-ddm/htool/actions/workflows/CI.yml) [![codecov](https://codecov.io/gh/htool-ddm/htool/branch/main/graph/badge.svg?token=1JJ40GPFA5)](https://codecov.io/gh/htool-ddm/htool)

## What is Htool?

Htool is an implementation of hierarchical matrices (cf. this [reference](http://www.springer.com/gp/book/9783662473238) or this [one](http://www.springer.com/gp/book/9783540771463)), it was written to test Domain Decomposition Methods (DDM) applied to Boundary Element Method (BEM). It provides:

* routines to build hierarchical matrix structures (cluster trees, block trees, low-rank matrices and block matrices),
* parallel matrix-vector and matrix-matrix product using MPI and OpenMP,
* preconditioning techniques using domain decomposition methods,
* the possibility to use Htool with any generator of coefficients (e.g., your own BEM library),
* an interface with [HPDDM](https://github.com/hpddm/hpddm) for iterative solvers,
* and several utility functions to display information about matrix structures and timing.

It is now used in [FreeFEM](https://freefem.org) starting from version 4.5, and we developed a [Python interface](https://github.com/htool-ddm/htool_python) using `pybind11`.

## How to use Htool?

### Dependencies

Htool is a header-only library written in C++11 with MPI and OpenMP, but it can be used without the latter if needed. Then, to use Htool, it requires:

* BLAS, to perform algebraic operations (dense matrix-matrix or matrix-vector operations).

And if you want, it needs:

* LAPACK, to perform SVD compression,
* HPDDM and its dependencies (BLAS, LAPACK) to use iterative solvers and DDM preconditioners.


### Installing

Since Htool is a header-only library, you can just include in your code the `include` folder of this repository. You can also use the following command to install the `include` folder in the one of your system to make it more widely available: in the root of this repository on your system, you can do:

```bash
mkdir build
cd build
cmake ..
cmake --build . --config Release --target install -- -j $(nproc)
```

Note that you can modify the `install` prefix using `cmake ..  -DCMAKE_INSTALL_PREFIX:PATH=your/install/path` instead of the third line.

### Embedding Htool in your code

We mostly refer to `smallest_example.cpp` and `smallest_example.py` in the `examples` folder to see how to use Htool.

A function that generates the coefficients must be provided to Htool. To do so, a structure inheriting from `IMatrix<T>` must be defined with a method called `T get_coef(const int& i, const int& j) const`, where `T` is the type of your coefficients. This method will return the coefficient (i,j) of the considered problem. A method `get_submatrix` can also be defined to provide a more efficient way to build a sub-block of the matrix. An example of such interface is given in `test_hmat_partialACA.hpp` or [BemTool](https://github.com/xclaeys/BemTool) (see `bemtool/miscellaneous/htool_wrap.hpp`). This new structure and the geometry will be used to define an object `HMatrix`.

A new type of compressor can also be added by defining a structure inheriting from `LowRankMatrix` with a method called `build` which populates the data members needed (see `partialACA.hpp`).

### Python interface

See this [repository](https://github.com/htool-ddm/htool_python).

## Who is behind Htool?

If you need help or have questions regarding Htool, feel free to contact [Pierre Marchand](https://www.ljll.math.upmc.fr/marchandp/) and Pierre-Henri Tournier.

## Acknowledgements

[University of Bath](https://www.bath.ac.uk), United Kingdom  
[ANR NonlocalDD](https://www.ljll.math.upmc.fr/~claeys/nonlocaldd/index.html), (grant ANR-15-CE23-0017-01), France  
[Inria](http://www.inria.fr/en/) Paris, France  
[Laboratoire Jacques-Louis Lions](https://www.ljll.math.upmc.fr/en/) Paris, France  

## Collaborators/contributors

[Matthieu Ancellin](https://ancell.in)  
[Xavier Claeys](https://www.ljll.math.upmc.fr/~claeys/)  
[Pierre Jolivet](http://jolivet.perso.enseeiht.fr/)  
[Frédéric Nataf](https://www.ljll.math.upmc.fr/nataf/)

![ANR NonlocalDD](figures/anr_nonlocaldd.png)
