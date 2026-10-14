#ifndef HTOOL_HMATRIX_LINALG_OMP_ADD_HMATRIX_MATRIX_PRODUCT_ROW_MAJOR_HPP
#define HTOOL_HMATRIX_LINALG_OMP_ADD_HMATRIX_MATRIX_PRODUCT_ROW_MAJOR_HPP

#include "../../../misc/logger.hpp"
#include "../../../misc/misc.hpp"
#include "../../hmatrix.hpp"
#include "../seq/add_hmatrix_matrix_product_row_major.hpp"
#include <algorithm>
#include <string>
#include <vector>

namespace htool {

template <typename CoefficientPrecision, typename CoordinatePrecision = underlying_type<CoefficientPrecision>>
void openmp_internal_add_hmatrix_matrix_product_row_major(char transa, char transb, CoefficientPrecision alpha, const HMatrix<CoefficientPrecision, CoordinatePrecision> &A, const CoefficientPrecision *B, CoefficientPrecision beta, CoefficientPrecision *C, int mu) {
    std::vector<const HMatrix<CoefficientPrecision, CoordinatePrecision> *> leaves;
    std::vector<const HMatrix<CoefficientPrecision, CoordinatePrecision> *> leaves_for_symmetry;
    std::tie(leaves, leaves_for_symmetry) = get_leaves_from(A); // C++17 structured binding

    if ((transa == 'T' && A.get_symmetry_for_leaves() == 'H')
        || (transa == 'C' && A.get_symmetry_for_leaves() == 'S')) {
        htool::Logger::get_instance().log(LogLevel::ERROR, "Operation is not supported (transa=" + std::string(1, transa) + " with " + A.get_symmetry_for_leaves() + ")"); // LCOV_EXCL_LINE
    }
    if (transb != 'N') {
        htool::Logger::get_instance().log(LogLevel::ERROR, "Operation is not implemented for openmp_internal_add_hmatrix_matrix_product_row_major (transb=" + std::string(1, transb) + ")"); // LCOV_EXCL_LINE
    }

    int out_size(A.get_target_cluster().get_size() * mu);
    auto get_output_cluster{&HMatrix<CoefficientPrecision, CoordinatePrecision>::get_target_cluster};
    auto get_input_cluster{&HMatrix<CoefficientPrecision, CoordinatePrecision>::get_source_cluster};
    int local_output_offset = A.get_target_cluster().get_offset();
    int local_input_offset  = A.get_source_cluster().get_offset();
    char trans_sym          = (A.get_symmetry_for_leaves() == 'S') ? 'T' : 'C';

    if (transa != 'N') {
        out_size            = A.get_source_cluster().get_size() * mu;
        get_input_cluster   = &HMatrix<CoefficientPrecision, CoordinatePrecision>::get_target_cluster;
        get_output_cluster  = &HMatrix<CoefficientPrecision, CoordinatePrecision>::get_source_cluster;
        local_input_offset  = A.get_target_cluster().get_offset();
        local_output_offset = A.get_source_cluster().get_offset();
        trans_sym           = 'N';
    }

    int incx(1), incy(1);
    if (CoefficientPrecision(beta) != CoefficientPrecision(1)) {
        Blas<CoefficientPrecision>::scal(&out_size, &beta, C, &incx);
    }

    // Contribution champ lointain
#if defined(_OPENMP)
#    pragma omp parallel
#endif
    {
        std::vector<CoefficientPrecision> temp(out_size, 0);
#if defined(_OPENMP)
#    pragma omp for schedule(guided) nowait
#endif
        for (int b = 0; b < leaves.size(); b++) {
            int input_offset  = (leaves[b]->*get_input_cluster)().get_offset();
            int output_offset = (leaves[b]->*get_output_cluster)().get_offset();
            internal_add_hmatrix_matrix_product_row_major(transa, transb, CoefficientPrecision(1), *leaves[b], B + (input_offset - local_input_offset) * mu, CoefficientPrecision(1), temp.data() + (output_offset - local_output_offset) * mu, mu);
        }

        // Symmetry part of the diagonal part
        if (A.get_symmetry_for_leaves() != 'N') {
#if defined(_OPENMP)
#    pragma omp for schedule(guided) nowait
#endif
            for (int b = 0; b < leaves_for_symmetry.size(); b++) {
                int input_offset  = (leaves_for_symmetry[b]->*get_input_cluster)().get_offset();
                int output_offset = (leaves_for_symmetry[b]->*get_output_cluster)().get_offset();
                internal_add_hmatrix_matrix_product_row_major(trans_sym, 'N', CoefficientPrecision(1), *leaves_for_symmetry[b], B + (output_offset - local_input_offset) * mu, CoefficientPrecision(1), temp.data() + (input_offset - local_output_offset) * mu, mu);
            }
        }

#if defined(_OPENMP)
#    pragma omp critical
#endif
        Blas<CoefficientPrecision>::axpy(&out_size, &alpha, temp.data(), &incx, C, &incy);
    }
}

} // namespace htool

#endif
