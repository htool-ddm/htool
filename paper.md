---
title: 'Htool-DDM: A C++ library for parallel solvers and compressed linear systems.'
tags:
  - C++
  - DDM preconditioners
  - compression
  - hierarchical matrices
  - clustering
  - distributed solver
authors:
  - name: Pierre Marchand
    orcid: 0000-0002-2522-6837
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: 1
  - name: Pierre-Henri Tournier
    affiliation: 2
  - name: Pierre Jolivet
    affiliation: 3
affiliations:
 - name: POEMS, CNRS, Inria, ENSTA, Institut Polytechnique de Paris, 91120 Palaiseau, France
   index: 1
 - name: Sorbonne Université, Université Paris Cité, CNRS, Inria, Laboratoire Jacques-Louis Lions, LJLL, EPC ALPINES, 4 place Jussieu, Paris F-75005, France
   index: 2
 - name: Sorbonne Université, CNRS, LIP6, 75252 Paris, France
   index: 3
date: 13 September 2025
bibliography: paper.bib

---

# Summary

Compressible dense linear systems arise in many applications such as:

- discretization of boundary integral equations [@BoermGrasedyckEtAl2003IHM],
- solving Lyapunov and Riccati equations [@BoermGrasedyckEtAl2003IHM],
- discretization of the integral Fractional Laplacian [@AinsworthGlusa2018TEF], and
- kernel-based scattered data interpolation [@IskeLeBorneEtAl2017HMA].

They are often derived from a discretization of asymptotically smooth kernels $\kappa(\mathbf{x},\mathbf{y})$, i.e., for $\mathbf{x}\neq\mathbf{y}$
$$
    \rvert \partial_x^{\alpha} \partial_y^{\beta}\kappa (\mathbf{x},\mathbf{y})\lvert \leq C_{\mathrm{as}}\lvert \mathbf{x} - \mathbf{y}\rvert^{-\lvert \alpha \rvert -\lvert \beta \rvert - s},
$$
where $\alpha,\beta \in \mathbb{N}_0^d$, with $\alpha+\beta\neq 0$ and $C_{\mathrm{as}},s\in \mathbb{R}$ are constants.

This mathematical property implies that the interaction between two distant clusters can be represented by a low-rank matrix. Multiple compression techniques have been developed to take advantage of this low-rank structure and provide an approximated representation of the a priori dense linear systems, for example, $\mathcal{H}$-matrix and $\mathcal{H}^2$-matrix [@Hackbusch2015HMA; @Bebendorf2008HMM; @Borm2010ENM] and FMM [@GreengardRokhlin1987FAP]. The advantages are usually:

- to avoid the assembly of the full dense linear system, and
- to provide a compressed/approximated version of all or part of the usual dense linear algebra.

The goal of such compression techniques is to lower the memory footprint and the cost of solving the linear system. For the latter, they provide an approximated matrix-vector product whose complexity scales linearly or quasi-linearly with the size of the problem. Thus, iterative solvers such as Conjugate Gradient or GMRES are well-suited.

Beside compression, another technique to accelerate iterative linear solvers is to use preconditioners: instead of solving $\mathbf{A}\mathbf{x}=\mathbf{f}$, where $\mathbf{A}$ is a dense or compressed linear system, we solve $\mathbf{P}\mathbf{A}\mathbf{x}=\mathbf{P}\mathbf{f}$ where $\mathbf{P}$ should be relatively inexpensive to apply, and $\mathbf{P}\mathbf{A}$ more suitable for an iterative resolution.

A class of preconditioners stemming from domain decomposition methods (DDM) are *Schwarz preconditioners*. They rely on a decomposition with overlap of the initial domain. $\mathbf{P}$ is constructed by solving well-chosen problems within each subdomain, together with a small global problem, called coarse space, that provides the necessary global information for the method to retain robustness as the number of subdomains increases. Such preconditioners have been introduced for boundary integral equations in @Hebeker1990PSA and analyzed in @StephanTran1998DDA and @Heuer1996EA.

One goal of `Htool-DDM` is to provide black-box DDM solvers using an adaptive coarse space called GenEO for "Generalized Eigenproblems problem in the Overlap" [@SpillaneDoleanEtAl2013ARC]. It has been adapted for specific boundary integral problems in @MarchandClaeysEtAl2020TLP and @Marchand2020SMB. The library also includes a default black-box matrix compression via an in-house $\mathcal{H}$-matrix implementation. `Htool-DDM` is a flexible platform to use and develop new DDM preconditioners and explore compression techniques via its multiple customization points.

# Statement of need

`Htool-DDM` is a lightweight C++ library that provides an easy-to-use interface for distributed iterative solvers and standard black-box matrix compression techniques via an in-house $\mathcal{H}$-matrix implementation. Its goal is to provide DDM preconditioners for dense/compressed linear systems.

It has many customization points to support research on efficient DDM preconditioners and compression techniques. For example, the user can provide their own compression algorithm, or customize the default hierarchical compression. Via its interface with HPDDM from @JolivetHechtEtAl2013SDD, it is also a flexible tool to test various iterative solvers and preconditioners, where local and/or global problems associated with DDM preconditioners can be tailored to the problem at hand.

The library has four main components:

- `Cluster` contains a hierarchical partition of a geometry. Various partition strategies are available, and users can define their own. See Figure \ref{fig:cluster_tree} and [examples/use_clustering.cpp](https://github.com/htool-ddm/htool/blob/d7c0fa8b42c461446b92a4891dea78532eeea6b5/examples/use_clustering.cpp).

![Level 2 of cluster (quad)tree for rotated ellipse.](figures/Figure_1.png){#fig:cluster_tree width="70%"}

- `HMatrix` represents a compressed kernel using $\mathcal{H}$-matrix based on:
    - a user-defined function that takes row and column indexes and generates the associated subblock of the matrix to be compressed; and
    - cluster trees contained in `Cluster` objects and representing the hierarchical partition of the underlying geometry on which the kernel is applied.

  Most linear algebra is supported with shared-memory parallelism and an interface inspired by [BLAS](https://www.netlib.org/blas/)/[LAPACK](https://www.netlib.org/lapack/) and [std::linalg](https://en.cppreference.com/w/cpp/numeric/linalg.html). Basic compression and recompression techniques are available, and users can also provide their own. See Figure \ref{fig:hmatrix} and [examples/use_hmatrix.cpp](https://github.com/htool-ddm/htool/blob/d7c0fa8b42c461446b92a4891dea78532eeea6b5/examples/use_hmatrix.cpp).

![Symmetric $\mathcal{H}$-matrix for $\frac{1}{1e^{-5}+\lvert x-y\rvert}$. Red blocks are dense, numbers in green blocks correspond to rank.](figures/Figure_2.png){#fig:hmatrix width="70%"}

- `DistributedOperator` is an MPI-distributed linear operator with a matrix-vector/matrix-matrix product. By default, it is defined as row-wise distributed operator where it uses locally a `HMatrix` for compression. It can also be interfaced with an external compression provider. See [examples/use_distributed_operator.cpp](https://github.com/htool-ddm/htool/blob/d7c0fa8b42c461446b92a4891dea78532eeea6b5/examples/use_distributed_operator.cpp).
- `DDM` models an iterative solver based on a `DistributedOperator` and HPDDM [@JolivetHechtEtAl2013SDD]. It also deals with the assembly and application of DDM preconditioners. The latter can be built from local problems solved with $\mathcal{H}$-LU factorization, for example. See [examples/use_ddm_solver.cpp](https://github.com/htool-ddm/htool/blob/d7c0fa8b42c461446b92a4891dea78532eeea6b5/examples/use_ddm_solver.cpp).




Examples of libraries related to compression are `H2lib` [@h2lib], `hmat-oss` [@hmat_oss], `HODLRlib` [@AmbikasaranSinghEtAl2019HLH], and H2Opus [@ZampiniBoukaramEtAl2022HDM]. One major distinction of `Htool-DDM` from these libraries is its focus on DDM preconditioners, where the in-house compression is only one component that can actually be replaced or used in conjunction with such external libraries. It should also be noted that these libraries often provide the discretization of the problem at hand directly (typically a finite element method for boundary integral equations). This is not the case of `Htool-DDM` where the interface is kept algebraic.

The library has been used for the numerical experiments in @MarchandClaeysEtAl2020TLP and @Marchand2020SMB. It has been included in FreeFEM ([@Hecht2012NDF], version $\geq$ 4.5) to support boundary integral equations, and in PETSc ([@BalayAbhyankarEtAl2020PUM], version $\geq$ 3.16) for black-box compression and DDM solvers using the [MatHtool](https://petsc.org/main/manualpages/Mat/MATHTOOL/) PETSc matrix type and seamless integration into PCHPDDM [@JolivetRomanEtAl2021KPE]. It also has its own [Python interface](https://github.com/htool-ddm/htool_python).

We refer to the [documentation](https://htool-ddm.pages.math.cnrs.fr) for more details.

# Acknowledgements

We acknowledge recent contributions from Virgile Dubos and scientific discussions with Igor Chollet, Xavier Claeys, Luiz M. Faria, Christian Glusa, Zoïs Moitier, and Frédéric Nataf that shaped the current version of `Htool-DDM`.

# References
