#ifndef HTOOL_HMATRIX_LINALG_ADD_HMATRIX_MATRIX_PRODUCT_ROW_MAJOR_HPP
#define HTOOL_HMATRIX_LINALG_ADD_HMATRIX_MATRIX_PRODUCT_ROW_MAJOR_HPP

#include "../hmatrix.hpp"

namespace htool {

template <typename CoefficientPrecision, typename CoordinatePrecision = CoefficientPrecision>
void sequential_add_hmatrix_matrix_product_row_major(char transa, char transb, CoefficientPrecision alpha, const HMatrix<CoefficientPrecision, CoordinatePrecision> &A, const CoefficientPrecision *B, CoefficientPrecision beta, CoefficientPrecision *C, int mu) {
    // set_leaves_in_cache();
    auto &leaves              = A.get_leaves();
    auto &leaves_for_symmetry = A.get_leaves_for_symmetry();

    if ((transa == 'T' && A.get_symmetry_for_leaves() == 'H')
        || (transa == 'C' && A.get_symmetry_for_leaves() == 'S')) {
        htool::Logger::get_instance().log(LogLevel::ERROR, "Operation is not supported (transa=" + std::string(1, transa) + " with " + A.get_symmetry_for_leaves() + ")"); // LCOV_EXCL_LINE
    }
    if (transb != 'N') {
        htool::Logger::get_instance().log(LogLevel::ERROR, "Operation is not implemented for sequential_add_hmatrix_matrix_product_row_major (transb=" + std::string(1, transb) + ")"); // LCOV_EXCL_LINE
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
        // TODO: use blas
        std::transform(C, C + out_size, C, [&beta](CoefficientPrecision &c) { return c * beta; });
    }

    // Contribution champ lointain
    std::vector<CoefficientPrecision> temp(out_size, 0);
    for (int b = 0; b < leaves.size(); b++) {
        int input_offset  = (leaves[b]->*get_input_cluster)().get_offset();
        int output_offset = (leaves[b]->*get_output_cluster)().get_offset();
        leaves[b]->add_matrix_product_row_major(transa, 1, B + (input_offset - local_input_offset) * mu, 1, temp.data() + (output_offset - local_output_offset) * mu, mu);
    }

    // Symmetry part of the diagonal part
    if (A.get_symmetry_for_leaves() != 'N') {
        for (int b = 0; b < leaves_for_symmetry.size(); b++) {
            int input_offset  = (leaves_for_symmetry[b]->*get_input_cluster)().get_offset();
            int output_offset = (leaves_for_symmetry[b]->*get_output_cluster)().get_offset();
            leaves_for_symmetry[b]->add_matrix_product_row_major(trans_sym, 1, B + (output_offset - local_input_offset) * mu, 1, temp.data() + (input_offset - local_output_offset) * mu, mu);
        }
    }
    Blas<CoefficientPrecision>::axpy(&out_size, &alpha, temp.data(), &incx, C, &incy);
}

template <typename CoefficientPrecision, typename CoordinatePrecision = CoefficientPrecision>
void openmp_add_hmatrix_matrix_product_row_major(char transa, char transb, CoefficientPrecision alpha, const HMatrix<CoefficientPrecision, CoordinatePrecision> &A, const CoefficientPrecision *B, CoefficientPrecision beta, CoefficientPrecision *C, int mu) {
    // set_leaves_in_cache();
    auto &leaves              = A.get_leaves();
    auto &leaves_for_symmetry = A.get_leaves_for_symmetry();

    if ((transa == 'T' && A.get_symmetry_for_leaves() == 'H')
        || (transa == 'C' && A.get_symmetry_for_leaves() == 'S')) {
        htool::Logger::get_instance().log(LogLevel::ERROR, "Operation is not supported (transa=" + std::string(1, transa) + " with " + A.get_symmetry_for_leaves() + ")"); // LCOV_EXCL_LINE
    }
    if (transb != 'N') {
        htool::Logger::get_instance().log(LogLevel::ERROR, "Operation is not implemented for openmp_add_hmatrix_matrix_product_row_major (transb=" + std::string(1, transb) + ")"); // LCOV_EXCL_LINE
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
        // TODO: use blas
        std::transform(C, C + out_size, C, [&beta](CoefficientPrecision &c) { return c * beta; });
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
            add_hmatrix_matrix_product_row_major(transa, transb, CoefficientPrecision(1), *leaves[b], B + (input_offset - local_input_offset) * mu, CoefficientPrecision(1), temp.data() + (output_offset - local_output_offset) * mu, mu);
        }

        // Symmetry part of the diagonal part
        if (A.get_symmetry_for_leaves() != 'N') {
#if defined(_OPENMP)
#    pragma omp for schedule(guided) nowait
#endif
            for (int b = 0; b < leaves_for_symmetry.size(); b++) {
                int input_offset  = (leaves_for_symmetry[b]->*get_input_cluster)().get_offset();
                int output_offset = (leaves_for_symmetry[b]->*get_output_cluster)().get_offset();
                add_hmatrix_matrix_product_row_major(trans_sym, 'N', CoefficientPrecision(1), *leaves_for_symmetry[b], B + (output_offset - local_input_offset) * mu, CoefficientPrecision(1), temp.data() + (input_offset - local_output_offset) * mu, mu);
            }
        }

#if defined(_OPENMP)
#    pragma omp critical
#endif
        Blas<CoefficientPrecision>::axpy(&out_size, &alpha, temp.data(), &incx, C, &incy);
    }
}

template <typename CoefficientPrecision, typename CoordinatePrecision>
void add_hmatrix_matrix_product_row_major(char transa, char transb, CoefficientPrecision alpha, const HMatrix<CoefficientPrecision, CoordinatePrecision> &A, const CoefficientPrecision *in, CoefficientPrecision beta, CoefficientPrecision *out, int mu) {
    switch (A.get_storage_type()) {
    case HMatrix<CoefficientPrecision, CoordinatePrecision>::StorageType::Dense:
        if (A.get_symmetry() == 'N') {
            A.get_dense_data()->add_matrix_product_row_major(transa, alpha, in, beta, out, mu);
        } else {
            A.get_dense_data()->add_matrix_product_symmetric_row_major(transa, alpha, in, beta, out, mu, A.get_UPLO(), A.get_symmetry());
        }
        break;
    case HMatrix<CoefficientPrecision, CoordinatePrecision>::StorageType::LowRank:
        A.get_low_rank_data()->add_matrix_product_row_major(transa, alpha, in, beta, out, mu);
        break;
    default:
        openmp_add_hmatrix_matrix_product_row_major(transa, transb, alpha, A, in, beta, out, mu);
        break;
    }
}
} // namespace htool

#endif
