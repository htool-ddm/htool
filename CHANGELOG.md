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

- `set_delay_dense_computation` to `HMatrix`

### Changed

### Fixed

- Fix const-correctness for g++ 4.8.5

## [0.8] - 2022-01-27

### Added

- doxygen documentation
- mvprod_transp_global_to_global and mvprod_transp_local_to_local added to VirtualHMatrix
- getters for clusters in VirtualHMatrix
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
- VirtualLowRankGenerator and VirtualAdmissibilityCondition added for better modularity
  
### Changed

- Remove unnecessary arguments in HMatrix and cluster interfaces
- MutliHMatrix deprecated for the moment (everything related to this in `htool/multi`)

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
