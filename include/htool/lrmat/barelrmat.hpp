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
template <typename T>
class MultipartialACA;

template <typename T>
class bareLowRankMatrix final : public LowRankMatrix<T> {
  private:
    // Friend
    friend class MultipartialACA<T>;

  public:
    using LowRankMatrix<T>::LowRankMatrix;

    void build(const VirtualGenerator<T> &A, const VirtualCluster &t, const double *const xt, const int *const tabt, const VirtualCluster &s, const double *const xs, const int *const tabs) {}
};
} // namespace htool
#endif
