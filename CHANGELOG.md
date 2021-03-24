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

## [Unreleased]

### Added

- `Cluster` class and its derivative `NCluster` with different splitting choices (`geometric_splitting` and `regular_splitting`)
- `Block` class with different admissible condition used in `Block` (only `RjasanowSteinbach` at the moment)
- `LowRankMatrix` and its derivatives (`SVD`, `fullACA`, `partialACA`, and `sympartialACA`)
- `IMatrix` class and its derivatives `Matrix` and `SubMatrix`
- `HMatrix` class and basic linear algebra (hmatrix-vector product, hmatrix-matrix product)
- `DDM` solvers via HPDDM, and coarse space building
