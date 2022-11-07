#ifndef HTOOL_VIRTUAL_LRMAT_GENERATOR_HPP
#define HTOOL_VIRTUAL_LRMAT_GENERATOR_HPP

#include <cassert>
#include <iterator>

namespace htool {

template <typename T>
class VirtualLowRankGenerator {
  public:
    VirtualLowRankGenerator() {}

    // C style
    virtual void copy_low_rank_approximation(double epsilon, int M, int N, const int *const rows, const int *const cols, int &rank, T **U, T **V, const VirtualGenerator<T> &A, const VirtualCluster &t, const double *const xt, const VirtualCluster &s, const double *const xs) const = 0;

    virtual bool is_htool_owning_data() const { return true; }
    virtual ~VirtualLowRankGenerator(){};
};

} // namespace htool

#endif
