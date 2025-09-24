<!--USE THIS TEMPLATE TO COMPLETE THE CHANGELOG-->
<!--
## [Version number] - YYYY-MM-DD
### Added
-

### Changed
-

### Deprecated
-

### Removed
-

### Fixed
-

### Security
-
-->

# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

### Added

- `HMatrix` recompression with SVD.
- Generic recompressed low-rank compression with `RecompressedLowRankGenerator`.
- Checks about `UPLO` for hmatrix factorization.
- `HMatrixBuilder` for easier `HMatrix` creation (especially when using only the hmatrix part of Htool-DDM).
- `add_hmatrix_vector_product` and `add_hmatrix_matrix_product` for working in user numbering. For C++17 and onward, these functions have preliminary support for execution policies with default being sequential execution. This tries to some extent to follow `<linalg>` API.
- `task_based_tree_builder.hpp` for miscellaneous functions used for task based approach.
- `hmatrix_output_dot.hpp` for L0 and block tree visualization.
- `task_based_compute_blocks` for task based alternative to `compute_blocks`.
- `task_based_internal_add_hmatrix_vector_product` for task based alternative to `internal_add_hmatrix_vector_product`.
- `task_based_internal_add_hmatrix_hmatrix_product` for task based alternative to `internal_add_hmatrix_hmatrix_product`.
- `task_based_internal_triangular_hmatrix_hmatrix_solve` for task based alternative to `internal_triangular_hmatrix_hmatrix_solve`.
- `task_based_lu_factorization` and `task_based_cholesky_factorization` for task based alternatives to `lu_factorization` and `cholesky_factorization`.
- `test_task_based_hmatrix_***.hpp` for testing various task based features.
- `internal_add_lrmat_hmatrix` is now overloaded to handle the case where the HMatrix is larger than the LowRankMatrix.
- `get_leaves_from` is overloaded to return non const arguments.
- `get_false_positive` and `get_L0` in a tree builder.
- `left_hmatrix_ancestor_of_right_hmatrix` and `left_hmatrix_descendant_of_right_hmatrix` for returning parent and children of a hmatrix.

### Changed

- `VirtualInternalLowRankGenerator` and `VirtualLowRankGenerator`'s `copy_low_rank_approximation` function takes a `LowRankMatrix` as input to populate it and returns a boolean. The return value is true if the compression succeded, false otherwise.
- `LowRankMatrix` constructors changed. It only takes sizes and an epsilon or a required rank. Then, it is expected to call a `VirtualInternalLowRankGenerator` to populate it.
- `ClusterTreeBuilder` has now one strategy as `VirtualPartitioning`. Usual implementations are still available, for example using `Partitioning<double,ComputeLargestExtent,RegularSplitting>`.
- When using `ClusterTreeBuilder` with `number_of_children=2^spatial_dimension`, it will do a binary/quad/octo-tree instead of `number_of_children` cut along the main direction.
- `ClusterTreeBuilder` parameter `minclustersize` was removed, and a parameter `maximal_leaf_size` has been added.
- `build` can now use `task_based_compute_blocks`

### Fixed

- Fix inline definition of `logging_level_to_string`.
- Fix error when resizing `Matrix`.
- Fix error due to using `int` instead of `size_t`, thanks to @vdubos.
- Fix warnings with `-Wold-style-cast`.

## [0.9.0] - 2024-09-19

### Added

- The old implementation of `HMatrix` was mixing the distributed operations and compression via hierarchical matrices. This is fixed by replacing `HMatrix` by:
    - `DistributedOperator` which contains a list of local operators and implements all the distributed operations,
    - `VirtualLocalOperator` which is the interface local operators must satisfy,
    - `LocalDenseMatrix` is an example of local operator consisting of a dense matrix `Matrix`,
    - and `LocalHMatrix` is an example of local operator consisting of a hierarchical matrix based on `HMatrix` (different from the previous `HMatrix`, see below).
- Utility classes that help build `DistributedOperator` and `DDM` objects are available, for example: `DefaultApproximationBuilder` and `DDMSolverBuilder`. They do all the wiring between the inner interfaces between Htool-DDM's objects, see `include/htool/istributed_operator/utility.hpp` and `include/htool/solvers/utility.hpp`.
- Formatter has been added, see `.clang_format`.
- A logger has been added with `Logger`. Its output can be customizerd via `IObjectWriter`.

### Changed

- `HMatrix` is now a class representing a hierarchical matrix without distributed-memory parallelism (note that it can still use shared-memory parallelism):
    - It inherits from `TreeNode`, and it provides the algebra related to hierarchical matrices via free functions:
        - product with vector and matrix (threaded with OpenMP),
        - and with this new version, LU and Cholesky factorization (not threaded yet, WIP).
    - The algorithms for building the block cluster tree is contained in `HMatrixTreeBuilder`. Users can provide their own "factory".
- `VirtualCluster` is removed and the clustering part of the library has been rewritten:
    - `Cluster` now derives from `TreeNode`, whose template parameter corresponds to the precision of cluster nodes' radius and centre (previously only `double`).
    - Standards recursive build algorithms are provided via `ClusterTreeBuilder`. Users can provide their own "factory".
    - `ClusterTreeBuilder` is a class template and uses the policy pattern (a policy for computing direction, and another for splitting along the direction).
- `DDM` has been modified, one-level and two-level preconditioners can be customized now via `VirtualLocalSolver`, `VirtualCoarseSpaceBuilder` and `VirtualCoarseOperatorBuilder`.
    - Three one-level preconditioners are provided by Htool-DDM via `DDMSolverBuilder`:
        - block-jacobi with a local HMatrix
        - a DDM preconditioner where the local subdomain with overlap uses one HMatrix
        - a DDM preconditioner where the local subdomain with overlap uses one HMatrix for the local subdomain without overlap, and dense matrices for the overlap and its interaction with the subdomain without overlap.
    - A second-level (GenEO coarse space) is provided with `GeneoCoarseSpaceDenseBuilder` and `GeneoCoarseOperatorBuilder`.

### Removed

- `multilrmat` and `multihmatrix` are removed.

## [0.8.1] - 2023-05-26

### Added

- `set_delay_dense_computation` to `HMatrix`
- clustering with user partition requiring permutation

### Changed

- improved CI with stricter flags
- moved version number definition from `misc/define.hpp` to `htool_version.hpp`

### Fixed

- Fix const-correctness for g++ 4.8.5
- Fix compatibility with c++20
- Fix MPI data type for complex float and double
- Fix issue with default MPI communicator

## [0.8] - 2022-01-27

### Added

- doxygen documentation
- mvprod_transp_global_to_global and mvprod_transp_local_to_local added to `VirtualHMatrix`
- getters for clusters in `VirtualHMatrix`
- custom gmv in ddm

### Changed

### Fixed

- Corner case with ACA resolved (first row/column only contains zeroes)
- Warnings from fujitsu compilers because of last line
- bug when using threading in mvprod

## [0.7] - 2021-09-21

### Added

- CMakefile checks consistency of version number across git/c++/CMake
- Test for warnings coming from `include/htool/*`
- Coverage added
- Methods in ddm interface to get local numbering
- `VirtualLowRankGenerator` and `VirtualAdmissibilityCondition` added for better modularity
  
### Changed

- Remove unnecessary arguments in HMatrix and cluster interfaces
- `MultiHMatrix` deprecated for the moment (everything related to this in `htool/multi`)

### Fixed

- Missing inlines added
- Fix bug when using htool via petsc with mkl

## [0.6] - 2021-08-03

### Changed

- Python interface with ctypes deprecated, see new python [interface](https://github.com/htool-ddm/htool_python) with pybind11.
- Performance tests removed, see dedicated [repository](https://github.com/PierreMarchand20/htool_benchmarks).
- GUI deprecated.

## [0.5] - 2021-08-03

### Added

- Interface for clustering via `VirtualCluster`, implementation with `Cluster` whose template parameter is the type of clustering (supported now: `PCARegularClustering`, `PCAGeometricClustering`, `BoundingBox1RegularClustering` and `BoundingBoxGeometricClustering`).
- Interface for hmatrix objects via `VirtualHMatrix`, implementation with HMatrix whose template parameter is the type of compressor (supported now: `SVD`, `fullACA`, `partialACA`, `sympartialACA`, all derived from abstract class `LowRankMatrix`).
- Interface for generator via `VirtualGenerator`. The user needs to define an object deriving from `VirtualGenerator` to populate a hmatrix.
- `Block` class with different admissible condition used in `Block` (only `RjasanowSteinbach` at the moment).
- Interface for a sub-block via `IMatrix`, from which both `SubMatrix` and `LowRankMatrix` inherit. They respectively correspond to dense and compressed sub-blocks.
- `DDM` solvers via HPDDM, and coarse space building.
- Test suite and CI with GitHub action.
