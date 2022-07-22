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
    int row_dimension;
    int column_dimension;

  public:
    VirtualGenerator(int nr0, int nc0, int row_dimension0 = 1, int column_dimension0 = 1) : nr(nr0), nc(nc0), row_dimension(row_dimension0), column_dimension(column_dimension0) {}

    // C style
    virtual void copy_submatrix(int M, int N, const int *const rows, const int *const cols, T *ptr) const = 0;

    int nb_rows() const { return nr; }
    int nb_cols() const { return nc; }
    int get_row_dimension() const { return row_dimension; }
    int get_column_dimension() const { return column_dimension; }

    virtual ~VirtualGenerator(){};
};

} // namespace htool

#endif
