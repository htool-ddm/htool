#ifndef HTOOL_MATRIX_LINALG_TRANSPOSE_HPP
#define HTOOL_MATRIX_LINALG_TRANSPOSE_HPP

#include "../../matrix/matrix.hpp" // for Matrix

namespace htool {

template <typename T>
void transpose(const Matrix<T> &A, Matrix<T> &B) {
    for (int i = 0; i < A.nb_rows(); i++) {
        for (int j = 0; j < A.nb_cols(); j++) {
            B(j, i) = A(i, j);
        }
    }
}

} // namespace htool
#endif
