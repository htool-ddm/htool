#ifndef HTOOL_GENERATOR_HPP
#define HTOOL_GENERATOR_HPP

#include "vector.hpp"
#include <cassert>
#include <iterator>

namespace htool {

template <typename T>
class VirtualGenerator {
  protected:
    // Data members
    int nr;
    int nc;
    int dimension;

  public:
    VirtualGenerator(int nr0, int nc0, int dimension0 = 1) : nr(nr0), nc(nc0), dimension(dimension0) {}

    // C style
    virtual void copy_submatrix(int M, int N, const int *const rows, const int *const cols, T *ptr) const = 0;

    int nb_rows() const { return nr; }
    int nb_cols() const { return nc; }
    int get_dimension() const { return dimension; }

    virtual ~VirtualGenerator(){};
};

} // namespace htool

#endif
