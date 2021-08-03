# Notes about the implementation

We rely heavily on polymorphism to make the library easy to extend in terms of functionality.

## Hierarchical matrix

`HMatrix` follows a [policy-based design](https://en.wikipedia.org/wiki/Modern_C%2B%2B_Design#Policy-based_design) (strategy pattern compile time variant), where the policies are

- the type of compression
- the type of admissible condition

The first policy is made explicit by the abstract class `LowRankMatrix`. This class defines the interface used by `HMatrix`, meaning that it has a function `build` that need to be implemented to define concrete compression. It allows adding new types of compression easily.

Note also that the `Cluster` is a shared resource with a shared pointer, so that it can be shared by several instances of `HMatrix`.

## Low-rank matrix

`LowRankMatrix` is an Abstract Base Class (ABC) that makes explicit the interface used by `HMatrix`. The pure virtual function is

```c++
virtual void build(const VirtualGenerator<T> &A, const VirtualCluster &t, const double *const xt, const int *const tabt, const VirtualCluster &s, const double *const xs, const int *const tabs) = 0;
```

If you want to add another type of compression, you need to define a derived class from `LowRankMatrix` that implement such a function. In this function you need to populate the data of a `LowRankMatrix`.

## Clustering

`HMatrix` needs cluster objects that follows the interface defined with `VirtualCluster`. In particular, Htool defines `Cluster` in `cluster.hpp`, which also follows a [policy-based design](https://en.wikipedia.org/wiki/Modern_C%2B%2B_Design#Policy-based_design). The policy is the type of clustering, for example `PCA`, which defines how the clusters are divided recursively. It needs to implement the following method:

```c++
void recursive_build(const double *const x, const double *const r, const int *const tab, const double *const g, int nb_sons, MPI_Comm comm, std::stack<Cluster<PCA> *> &s, std::stack<std::vector<int>> &n)
```
