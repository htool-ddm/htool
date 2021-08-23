#ifndef HTOOL_VIRTUAL_MULTI_LRMAT_GENERATOR_HPP
#define HTOOL_VIRTUAL_MULTI_LRMAT_GENERATOR_HPP

#include "../clustering/virtual_cluster.hpp"
#include "../types/multimatrix.hpp"
#include <cassert>
#include <iterator>
namespace htool {

template <typename T>
class VirtualMultiLowRankGenerator {
  public:
    VirtualMultiLowRankGenerator() {}

    // C style
    virtual void copy_multi_low_rank_approximation(double epsilon, int M, int N, const int *const rows, const int *const cols, int &rank, T ***U, T ***V, const MultiIMatrix<T> &A, const VirtualCluster &t, const double *const xt, const VirtualCluster &s, const double *const xs) const = 0;

    virtual ~VirtualMultiLowRankGenerator(){};
};

} // namespace htool

#endif
