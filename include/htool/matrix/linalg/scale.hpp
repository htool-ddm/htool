#ifndef HTOOL_MATRIX_LINALG_SCALE_HPP
#define HTOOL_MATRIX_LINALG_SCALE_HPP

#include "../../matrix/matrix.hpp"         // for Matrix
#include "../../wrappers/wrapper_blas.hpp" // for Blas

namespace htool {

template <typename T>
void scale(T da, Matrix<T> &A) {
    int size = A.nb_cols() * A.nb_rows();
    int incx = 1;
    Blas<T>::scal(&size, &da, A.data(), &incx);
}

} // namespace htool
#endif
