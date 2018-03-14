## HTOOL [![Build Status](https://travis-ci.org/PierreMarchand20/htool.svg?branch=master)](https://travis-ci.org/PierreMarchand20/htool.svg?branch=master)

#### What is Htool ?

Htool is an implementation of hierarchical matrices (cf. this [reference](http://www.springer.com/gp/book/9783662473238) or this [one](http://www.springer.com/gp/book/9783540771463)), it was written to test Domain Decomposition Methods (DDM) applied to Boundary Element Method (BEM). It provides:
* routines to build hierarchical matrix structures (cluster trees, block trees, low-rank matrices and block matrices),
* parallel matrix-vector and matrix-matrix product using MPI and OpenMP,
* preconditioning techniques using domain decomposition methods,
* the possibility to use Htool with any generator of coefficients (e.g., your own BEM library),
* an interface with [HPDDM](https://github.com/hpddm/hpddm) for iterative solvers,
* GUI and several service functions to display informations about matrix structures and timing.

#### How to use Htool ?
Htool is a header library written in C++11 with MPI and OpenMP, but it can be used without the latter if needed. Then, Htool needs to be linked against :
* BLAS, to perform algebraic operations (dense matrix-matrix or matrix-vector operations),
* HPDDM and its dependencies (BLAS, LAPACK and a direct solver like [MUMPS](http://mumps.enseeiht.fr/), [SuiteSparse](http://faculty.cse.tamu.edu/davis/suitesparse.html), [MKL PARDISO](https://software.intel.com/en-us/articles/intel-mkl-pardiso), or [PaStiX](http://pastix.gforge.inria.fr/)) to use iterative solvers and DDM preconditioners,
* Eigen, to use SVD compressors (to be modified),
* Nanogui and its dependency (use `git submodule` in this repository to use it, see `Tests_view`), to use the GUI.

In any case, a function that generates the coefficients must be provided to Htool. To do so, a structure inheriting from `IMatrix<T>` must be defined with a method called `T get_coef(const int& i, const int& j) const`, where `T` is the type of your coefficients. This method will return the coefficient (i,j) of the considered problem. A method `get_submatrix` can also be defined to provide a more efficient way to build a sub-block of the matrix. An example of such interface is given in `test_hmat_partialACA.hpp` or  [BemTool](https://github.com/xclaeys/BemTool) (see `bemtool/miscellaneous/htool_wrap.hpp`).

A new type of compressor can also be added by defining a structure inheriting from `LowRankMatrix` with a method called `build` which populates the data members needed (see `partialACA.hpp`).

#### Who is behind Htool?
If you need help or have questions regarding Htool, feel free to contact [Pierre Marchand](https://www.ljll.math.upmc.fr/marchandp/) and Pierre-Henri Tournier.

#### Acknowledgements
[ANR NonlocalDD](https://www.ljll.math.upmc.fr/~claeys/nonlocaldd/index.html), (grant ANR-15-CE23-0017-01), France  
[Inria](http://www.inria.fr/en/) Paris, France  
[Laboratoire Jacques-Louis Lions](https://www.ljll.math.upmc.fr/en/) Paris, France  

#### Collaborators/contributors
[Xavier Claeys](https://www.ljll.math.upmc.fr/~claeys/)  
[Pierre Jolivet](http://jolivet.perso.enseeiht.fr/)  
[Frédéric Nataf](https://www.ljll.math.upmc.fr/nataf/)

![ANR NonlocalDD](figures/anr_nonlocaldd.png)
