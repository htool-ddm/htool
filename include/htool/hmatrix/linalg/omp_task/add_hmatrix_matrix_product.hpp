#ifndef HTOOL_HMATRIX_LINALG_OMP_TASK_ADD_HMATRIX_MATRIX_PRODUCT_HPP
#define HTOOL_HMATRIX_LINALG_OMP_TASK_ADD_HMATRIX_MATRIX_PRODUCT_HPP

#include "../../../matrix/linalg/transpose.hpp"
#include "../../../matrix/matrix.hpp"
#include "../../../misc/misc.hpp"
#include "../../hmatrix.hpp"
#include "add_hmatrix_matrix_product_row_major.hpp"
#include <algorithm>
#include <vector>

namespace htool {

// Same as openmp_internal_add_hmatrix_matrix_product, with the tasks of task_based_internal_add_hmatrix_matrix_product_row_major,
// returning once they are done.
template <typename MatB,
          typename MatC,
          typename CoordinatePrecision = underlying_type<typename MatB::value_type>,
          typename                     = std::enable_if_t<
                                  std::is_same<typename MatB::value_type, typename MatC::value_type>::value>>
void task_based_internal_add_hmatrix_matrix_product(char transa, char transb, typename MatB::value_type alpha, const HMatrix<typename MatB::value_type, CoordinatePrecision> &A, const MatB &B, typename MatB::value_type beta, MatC &C, const std::vector<const HMatrix<typename MatB::value_type, CoordinatePrecision> *> &L0, const std::vector<const Cluster<CoordinatePrecision> *> &in_L0, const std::vector<const Cluster<CoordinatePrecision> *> &out_L0) {
    using CoefficientPrecision = typename MatB::value_type;
    Matrix<CoefficientPrecision> transposed_C(C.nb_cols(), C.nb_rows());
    transpose(C, transposed_C);
    if (transb == 'N') {
        Matrix<CoefficientPrecision> transposed_B(B.nb_cols(), B.nb_rows());
        transpose(B, transposed_B);
        task_based_internal_add_hmatrix_matrix_product_row_major(transa, alpha, A, transposed_B.data(), beta, transposed_C.data(), transposed_C.nb_rows(), L0, in_L0, out_L0);
#if defined(_OPENMP)
#    pragma omp taskwait
#endif
    } else {
        std::vector<CoefficientPrecision> buffer_B(transb == 'C' ? B.nb_cols() * B.nb_rows() : 0);
        if (transb == 'C') {
            std::copy(B.data(), B.data() + B.nb_cols() * B.nb_rows(), buffer_B.data());
            conj_if_complex(buffer_B.data(), buffer_B.size());
        }
        task_based_internal_add_hmatrix_matrix_product_row_major(transa, alpha, A, transb == 'C' ? buffer_B.data() : B.data(), beta, transposed_C.data(), transposed_C.nb_rows(), L0, in_L0, out_L0);
#if defined(_OPENMP)
#    pragma omp taskwait
#endif
    }
    transpose(transposed_C, C);
}

} // namespace htool

#endif
