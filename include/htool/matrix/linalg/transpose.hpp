#ifndef HTOOL_MATRIX_LINALG_TRANSPOSE_HPP
#define HTOOL_MATRIX_LINALG_TRANSPOSE_HPP

namespace htool {

template <typename MatA,
          typename MatB,
          typename = std::enable_if_t<
              std::is_same<typename MatA::value_type, typename MatB::value_type>::value>>
void transpose(const MatA &A, MatB &B) {
    for (int i = 0; i < A.nb_rows(); i++) {
        for (int j = 0; j < A.nb_cols(); j++) {
            B(j, i) = A(i, j);
        }
    }
}

} // namespace htool
#endif
