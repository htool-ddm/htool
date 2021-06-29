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

  public:
    VirtualGenerator(int nr0, int nc0) : nr(nr0), nc(nc0) {}

    // C style
    virtual void copy_submatrix(int M, int N, const int *const rows, const int *const cols, T *ptr) const = 0;

    //! ### Access to number of rows
    /*!
    Returns the number of rows of the input argument _A_.
    */
    int nb_rows() const { return nr; }

    //! ### Access to number of columns
    /*!
    Returns the number of columns of the input argument _A_.
    */
    int nb_cols() const { return nc; }

    virtual ~VirtualGenerator(){};
};

} // namespace htool

#endif
