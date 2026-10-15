#ifndef HTOOL_HMATRIX_LINALG_OMP_ADD_HMATRIX_MATRIX_PRODUCT_HPP
#define HTOOL_HMATRIX_LINALG_OMP_ADD_HMATRIX_MATRIX_PRODUCT_HPP

#include "../../../matrix/linalg/transpose.hpp"
#include "../../../matrix/matrix.hpp"
#include "../../../misc/misc.hpp"
#include "../../hmatrix.hpp"
#include "add_hmatrix_matrix_product_row_major.hpp"
#include <algorithm>
#include <vector>

namespace htool {

template <typename MatB,
          typename MatC,
          typename CoordinatePrecision = underlying_type<typename MatB::value_type>,
          typename                     = std::enable_if_t<
                                  std::is_same<typename MatB::value_type, typename MatC::value_type>::value>>
void openmp_internal_add_hmatrix_matrix_product(char transa, char transb, typename MatB::value_type alpha, const HMatrix<typename MatB::value_type, CoordinatePrecision> &A, const MatB &B, typename MatB::value_type beta, MatC &C) {
    using CoefficientPrecision = typename MatB::value_type;
    if (transb == 'N') {
        Matrix<CoefficientPrecision> transposed_B(B.nb_cols(), B.nb_rows()), transposed_C(C.nb_cols(), C.nb_rows());
        transpose(B, transposed_B);
        transpose(C, transposed_C);
        openmp_internal_add_hmatrix_matrix_product_row_major(transa, transb, alpha, A, transposed_B.data(), beta, transposed_C.data(), transposed_C.nb_rows());
        transpose(transposed_C, C);
    } else {
        Matrix<CoefficientPrecision> transposed_C(C.nb_cols(), C.nb_rows());
        transpose(C, transposed_C);
        std::vector<CoefficientPrecision> buffer_B(transb == 'C' ? B.nb_cols() * B.nb_rows() : 0);

        if (transb == 'C') {
            std::copy(B.data(), B.data() + B.nb_cols() * B.nb_rows(), buffer_B.data());
            conj_if_complex(buffer_B.data(), buffer_B.size());
        }

        openmp_internal_add_hmatrix_matrix_product_row_major(transa, 'N', alpha, A, transb == 'C' ? buffer_B.data() : B.data(), beta, transposed_C.data(), transposed_C.nb_rows());
        transpose(transposed_C, C);
    }
}

} // namespace htool

#endif
