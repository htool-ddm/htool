#ifndef HTOOL_ZERO_GENERATOR_HPP
#define HTOOL_ZERO_GENERATOR_HPP

#include "vector.hpp"
#include <cassert>
#include <iterator>

namespace htool {

template <typename T>
class ZeroGenerator : public VirtualGenerator<T> {
  protected:
    // Data members

  public:
    ZeroGenerator(int nr0, int nc0, int dimension0 = 1) : VirtualGenerator<T>(nr0, nc0, dimension0) {}

    // C style
    void copy_submatrix(int M, int N, const int *const, const int *const, T *ptr) const override {
        std::fill_n(ptr, M * N, T(0));
    };
};

} // namespace htool

#endif
