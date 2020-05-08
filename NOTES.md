# Notes about the implementation

We rely heavily on polymorphism to make the library easy to extend in terms of functionality.

## Hierarchical matrix

`HMatrix` follows a [policy-based design](https://en.wikipedia.org/wiki/Modern_C%2B%2B_Design#Policy-based_design) (strategy pattern compile time variant), where the policies are

- the type of compression
- the type of clustering

These policies are made explicit respectively by the abstract classes `LowRankMatrix` and `Cluster`. These two classes define the interface used by `HMatrix` and they are abstract, meaning that they have both a function `build` that need to be implemented to define concret compression and clustering techniques. It allows to add new types of compression and clustering easily.

Note also that the `Cluster` is a shared ressource with a shared pointer, so that it can be shared by several instance of `HMatrix`.

## Low rank matrix

`LowRankMatrix` is an Abstract Base Class (ABC) that makes explicit the interface used by `HMatrix`. The pure virtual function is

```c++
virtual void build(const IMatrix<T>& A, const Cluster& t, const std::vector<R3>& xt,const std::vector<int>& tabt, const Cluster& s, const std::vector<R3>& xs, const std::vector<int>& tabs) = 0;
```

If you want to add another type of compression, you need to define a derived class from `LowRankMatrix` that implement such a function. In this function you need to populate the data of a `LowRankMatrix`.


IMatrix -> runtime polymorphism, we could use CRTP to do static polymorphism, but is it worth it ?


## Clustering

The clustering interface used by HMatrix is defined by `Cluster` in `cluster.hpp`. Using CRTP, implementations of clustering techniques can be defined as derived classes of `Cluster` using themselves as template parameter of `Cluster`. Then, one can define

```c++
void build(const std::vector<R3>& x0, const std::vector<double>& r0,const std::vector<int>& tab0, const std::vector<double>& g0, int nb_sons = -1, MPI_Comm comm=MPI_COMM_WORLD)
```

to use this implementation automatically in `HMatrix`. If one needs other inputs to define a clustering technique, another build function can be used, but instances of clustering techniques not using this `build` function need to be given to `HMatrix` after building them externally.

## IMatrix