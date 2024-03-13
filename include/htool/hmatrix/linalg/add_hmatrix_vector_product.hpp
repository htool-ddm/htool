#ifndef HTOOL_HMATRIX_LINALG_ADD_HMATRIX_VECTOR_PRODUCT_HPP
#define HTOOL_HMATRIX_LINALG_ADD_HMATRIX_VECTOR_PRODUCT_HPP

#include "../hmatrix.hpp"

namespace htool {

// template <typename CoefficientPrecision, typename CoordinatePrecision = CoefficientPrecision>
// void sequential_add_hmatrix_vector_product(char trans, CoefficientPrecision alpha, const HMatrix<CoefficientPrecision, CoordinatePrecision> &A, const CoefficientPrecision *in, CoefficientPrecision beta, CoefficientPrecision *out) {

//     int out_size(A.get_target_cluster().get_size());
//     auto get_output_cluster{&HMatrix<CoefficientPrecision, CoordinatePrecision>::get_target_cluster};
//     auto get_input_cluster{&HMatrix<CoefficientPrecision, CoordinatePrecision>::get_source_cluster};
//     int local_input_offset  = A.get_source_cluster().get_offset();
//     int local_output_offset = A.get_target_cluster().get_offset();
//     char trans_sym          = (A.get_symmetry_for_leaves() == 'S') ? 'T' : 'C';

//     if (trans != 'N') {
//         out_size            = A.get_source_cluster().get_size();
//         get_input_cluster   = &HMatrix<CoefficientPrecision, CoordinatePrecision>::get_target_cluster;
//         get_output_cluster  = &HMatrix<CoefficientPrecision, CoordinatePrecision>::get_source_cluster;
//         local_input_offset  = A.get_target_cluster().get_offset();
//         local_output_offset = A.get_source_cluster().get_offset();
//         trans_sym           = 'N';
//     }

//     if (CoefficientPrecision(beta) != CoefficientPrecision(1)) {
//         std::transform(out, out + out_size, out, [&beta](CoefficientPrecision &c) { return c * beta; });
//     }

//     // Contribution champ lointain
//     std::vector<CoefficientPrecision> temp(out_size, 0);
//     // for (int b = 0; b < A.get_leaves().size(); b++) {
//     for (auto &leaf : A.get_leaves()) {
//         int input_offset  = (leaf->*get_input_cluster)().get_offset();
//         int output_offset = (leaf->*get_output_cluster)().get_offset();
//         leaf->add_vector_product(trans, 1, in + input_offset - local_input_offset, alpha, out + (output_offset - local_output_offset));
//     }

//     // Symmetry part of the diagonal part
//     if (A.get_symmetry_for_leaves() != 'N') {
//         // for (int b = 0; b < A.get_leaves_for_symmetry().size(); b++) {
//         for (auto &leaf_for_symmetry : A.get_leaves_for_symmetry()) {
//             int input_offset  = (leaf_for_symmetry->*get_input_cluster)().get_offset();
//             int output_offset = (leaf_for_symmetry->*get_output_cluster)().get_offset();
//             leaf_for_symmetry->add_vector_product(trans_sym, alpha, in + output_offset - local_input_offset, 1, out + (input_offset - local_output_offset));
//         }
//     }
// }

template <typename CoefficientPrecision, typename CoordinatePrecision = CoefficientPrecision>
void openmp_add_hmatrix_vector_product(char trans, CoefficientPrecision alpha, const HMatrix<CoefficientPrecision, CoordinatePrecision> &A, const CoefficientPrecision *in, CoefficientPrecision beta, CoefficientPrecision *out) {

    // A.set_leaves_in_cache();
    auto &leaves              = A.get_leaves();
    auto &leaves_for_symmetry = A.get_leaves_for_symmetry();

    int out_size(A.get_target_cluster().get_size());
    auto get_output_cluster{&HMatrix<CoefficientPrecision, CoordinatePrecision>::get_target_cluster};
    auto get_input_cluster{&HMatrix<CoefficientPrecision, CoordinatePrecision>::get_source_cluster};
    int local_input_offset  = A.get_source_cluster().get_offset();
    int local_output_offset = A.get_target_cluster().get_offset();
    char trans_sym          = (A.get_symmetry_for_leaves() == 'S') ? 'T' : 'C';

    if (trans != 'N') {
        out_size            = A.get_source_cluster().get_size();
        get_input_cluster   = &HMatrix<CoefficientPrecision, CoordinatePrecision>::get_target_cluster;
        get_output_cluster  = &HMatrix<CoefficientPrecision, CoordinatePrecision>::get_source_cluster;
        local_input_offset  = A.get_target_cluster().get_offset();
        local_output_offset = A.get_source_cluster().get_offset();
        trans_sym           = 'N';
    }

    int incx(1), incy(1);
    if (CoefficientPrecision(beta) != CoefficientPrecision(1)) {
        // TODO: use blas
        std::transform(out, out + out_size, out, [&beta](CoefficientPrecision &c) { return c * beta; });
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
            leaves[b]->add_vector_product(trans, 1, in + input_offset - local_input_offset, 1, temp.data() + (output_offset - local_output_offset));
        }

        // Symmetry part of the diagonal part
        if (A.get_symmetry_for_leaves() != 'N') {
#if defined(_OPENMP)
#    pragma omp for schedule(guided) nowait
#endif
            for (int b = 0; b < leaves_for_symmetry.size(); b++) {
                int input_offset  = (leaves_for_symmetry[b]->*get_input_cluster)().get_offset();
                int output_offset = (leaves_for_symmetry[b]->*get_output_cluster)().get_offset();
                leaves_for_symmetry[b]->add_vector_product(trans_sym, 1, in + output_offset - local_input_offset, 1, temp.data() + (input_offset - local_output_offset));
            }
        }

#if defined(_OPENMP)
#    pragma omp critical
#endif
        Blas<CoefficientPrecision>::axpy(&out_size, &alpha, temp.data(), &incx, out, &incy);
    }
}

template <typename CoefficientPrecision, typename CoordinatePrecision>
void add_hmatrix_vector_product(char trans, CoefficientPrecision alpha, const HMatrix<CoefficientPrecision, CoordinatePrecision> &A, const CoefficientPrecision *in, CoefficientPrecision beta, CoefficientPrecision *out) {
    switch (A.get_storage_type()) {
    case HMatrix<CoefficientPrecision, CoordinatePrecision>::StorageType::Dense:
        if (A.get_symmetry() == 'N') {
            A.get_dense_data()->add_vector_product(trans, alpha, in, beta, out);
        } else {
            A.get_dense_data()->add_vector_product_symmetric(trans, alpha, in, beta, out, A.get_UPLO(), A.get_symmetry());
        }
        break;
    case HMatrix<CoefficientPrecision, CoordinatePrecision>::StorageType::LowRank:
        A.get_low_rank_data()->add_vector_product(trans, alpha, in, beta, out);
        break;
    default:
        openmp_add_hmatrix_vector_product(trans, alpha, A, in, beta, out);
        break;
    }
}
} // namespace htool

#endif
