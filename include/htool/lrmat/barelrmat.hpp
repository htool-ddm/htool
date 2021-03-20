#ifndef HTOOL_BARELRMAT_HPP
#define HTOOL_BARELRMAT_HPP

#include "../multilrmat/multilrmat.hpp"
#include "lrmat.hpp"
#include <cassert>
#include <complex>
#include <fstream>
#include <iostream>
#include <vector>

// To be used with multilrmat
namespace htool {

// Forward declaration
template <typename T, typename ClusterImpl>
class MultipartialACA;

template <typename T, typename ClusterImpl>
class bareLowRankMatrix final : public LowRankMatrix<T, ClusterImpl> {
  private:
    // Friend
    friend class MultipartialACA<T, ClusterImpl>;

  public:
    using LowRankMatrix<T, ClusterImpl>::LowRankMatrix;

    void build(const IMatrix<T> &A, const Cluster<ClusterImpl> &t, const std::vector<R3> &xt, const std::vector<int> &tabt, const Cluster<ClusterImpl> &s, const std::vector<R3> &xs, const std::vector<int> &tabs) {}
};
} // namespace htool
#endif
