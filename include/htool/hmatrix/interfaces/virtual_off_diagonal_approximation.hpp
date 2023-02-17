#ifndef HTOOL_VIRTUAL_OFF_DIAGONAL_APPROXIMATION_HPP
#define HTOOL_VIRTUAL_OFF_DIAGONAL_APPROXIMATION_HPP

#include "vector.hpp"
#include <cassert>
#include <iterator>

namespace htool {

template <typename T>
class VirtualOffDiagonalApproximation {
  public:
    // use htool numbering
    // if mu==1 local offdiagonal part of A*in + out
    // if mu>1, transpose(in)*transpose(A) + out, where out is transposed (and will be "untransposed" later on)
    // In other words, out and in are row-major
    // This is to localize block of rhs

    // If IsUsingLocalVector returns true, in is the local vector, otherwise is the concatenated off diagonal vector
    virtual void mvprod_off_diagonal_to_local(const T *const in, T *const out, const int &mu) = 0;

    virtual void mvprod_subrhs_to_local(const T *const in, T *const out, const int &mu, const int &offset, const int &size) = 0;

    virtual bool IsUsingRowMajorStorage() { return true; };
    virtual bool IsUsingLocalVector() { return false; };
    virtual ~VirtualOffDiagonalApproximation(){};
};

} // namespace htool

#endif
