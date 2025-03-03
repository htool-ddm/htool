#ifndef HTOOL_HMATRIX_LINALG_ADD_HMATRIX_VECTOR_PRODUCT_HPP
#define HTOOL_HMATRIX_LINALG_ADD_HMATRIX_VECTOR_PRODUCT_HPP

#include "../../matrix/linalg/add_matrix_vector_product.hpp" // for add_mat...
#include "../../misc/misc.hpp"                               // for underly...
#include "../../wrappers/wrapper_blas.hpp"                   // for Blas
#include "../hmatrix.hpp"                                    // for HMatrix
#include "../lrmat/linalg/add_lrmat_vector_product.hpp"      // for add_lrm...
#include "execution_policies.hpp"
#include <algorithm> // for transform
#include <complex>   // for complex
#include <vector>    // for vector

namespace htool {

template <typename CoefficientPrecision, typename CoordinatePrecision = underlying_type<CoefficientPrecision>>
void internal_add_hmatrix_vector_product(char trans, CoefficientPrecision alpha, const HMatrix<CoefficientPrecision, CoordinatePrecision> &A, const CoefficientPrecision *in, CoefficientPrecision beta, CoefficientPrecision *out) {
    switch (A.get_storage_type()) {
    case HMatrix<CoefficientPrecision, CoordinatePrecision>::StorageType::Dense:
        if (A.get_symmetry() == 'N') {
            add_matrix_vector_product(trans, alpha, *A.get_dense_data(), in, beta, out);
        } else if (A.get_symmetry() == 'S') {
            add_symmetric_matrix_vector_product(A.get_UPLO(), alpha, *A.get_dense_data(), in, beta, out);
        }
        break;
    case HMatrix<CoefficientPrecision, CoordinatePrecision>::StorageType::LowRank:
        add_lrmat_vector_product(trans, alpha, *A.get_low_rank_data(), in, beta, out);
        break;
    default:
        sequential_internal_add_hmatrix_vector_product(trans, alpha, A, in, beta, out);
        break;
    }
}

template <typename CoefficientPrecision, typename CoordinatePrecision = underlying_type<CoefficientPrecision>>
void internal_add_hmatrix_vector_product(char trans, std::complex<CoefficientPrecision> alpha, const HMatrix<std::complex<CoefficientPrecision>, CoordinatePrecision> &A, const std::complex<CoefficientPrecision> *in, std::complex<CoefficientPrecision> beta, std::complex<CoefficientPrecision> *out) {
    switch (A.get_storage_type()) {
    case HMatrix<std::complex<CoefficientPrecision>, CoordinatePrecision>::StorageType::Dense:
        if (A.get_symmetry() == 'N') {
            add_matrix_vector_product(trans, alpha, *A.get_dense_data(), in, beta, out);
        } else if (A.get_symmetry() == 'S') {
            add_symmetric_matrix_vector_product(A.get_UPLO(), alpha, *A.get_dense_data(), in, beta, out);
        } else if (A.get_symmetry() == 'H') {
            add_hermitian_matrix_vector_product(A.get_UPLO(), alpha, *A.get_dense_data(), in, beta, out);
        }
        break;
    case HMatrix<std::complex<CoefficientPrecision>, CoordinatePrecision>::StorageType::LowRank:
        add_lrmat_vector_product(trans, alpha, *A.get_low_rank_data(), in, beta, out);
        break;
    default:
        sequential_internal_add_hmatrix_vector_product(trans, alpha, A, in, beta, out);
        break;
    }
}

template <typename CoefficientPrecision, typename CoordinatePrecision = underlying_type<CoefficientPrecision>>
void sequential_internal_add_hmatrix_vector_product(char trans, CoefficientPrecision alpha, const HMatrix<CoefficientPrecision, CoordinatePrecision> &A, const CoefficientPrecision *in, CoefficientPrecision beta, CoefficientPrecision *out) {

    int out_size(A.get_target_cluster().get_size());
    auto get_output_cluster{&HMatrix<CoefficientPrecision, CoordinatePrecision>::get_target_cluster};
    auto get_input_cluster{&HMatrix<CoefficientPrecision, CoordinatePrecision>::get_source_cluster};
    int local_input_offset  = A.get_source_cluster().get_offset();
    int local_output_offset = A.get_target_cluster().get_offset();
    char trans_sym          = (A.get_symmetry_for_leaves() == 'S') ? 'T' : 'C';
    std::vector<const HMatrix<CoefficientPrecision, CoordinatePrecision> *> leaves;
    std::vector<const HMatrix<CoefficientPrecision, CoordinatePrecision> *> leaves_for_symmetry;
    std::tie(leaves, leaves_for_symmetry) = get_leaves_from(A); // C++17 structured binding

    if (trans != 'N') {
        out_size            = A.get_source_cluster().get_size();
        get_input_cluster   = &HMatrix<CoefficientPrecision, CoordinatePrecision>::get_target_cluster;
        get_output_cluster  = &HMatrix<CoefficientPrecision, CoordinatePrecision>::get_source_cluster;
        local_input_offset  = A.get_target_cluster().get_offset();
        local_output_offset = A.get_source_cluster().get_offset();
        trans_sym           = 'N';
    }

    if (CoefficientPrecision(beta) != CoefficientPrecision(1)) {
        std::transform(out, out + out_size, out, [&beta](CoefficientPrecision &c) { return c * beta; });
    }

    // Contribution champ lointain
    std::vector<CoefficientPrecision> temp(out_size, 0);
    for (auto &leaf : leaves) {
        int input_offset  = (leaf->*get_input_cluster)().get_offset();
        int output_offset = (leaf->*get_output_cluster)().get_offset();
        internal_add_hmatrix_vector_product(trans, alpha, *leaf, in + input_offset - local_input_offset, CoefficientPrecision(1), out + (output_offset - local_output_offset));
    }

    // Symmetry part of the diagonal part
    if (A.get_symmetry_for_leaves() != 'N') {
        for (auto &leaf_for_symmetry : leaves_for_symmetry) {
            int input_offset  = (leaf_for_symmetry->*get_input_cluster)().get_offset();
            int output_offset = (leaf_for_symmetry->*get_output_cluster)().get_offset();
            internal_add_hmatrix_vector_product(trans_sym, alpha, *leaf_for_symmetry, in + output_offset - local_input_offset, CoefficientPrecision(1), out + (input_offset - local_output_offset));
        }
    }
}

template <typename CoefficientPrecision, typename CoordinatePrecision = underlying_type<CoefficientPrecision>>
void openmp_internal_add_hmatrix_vector_product(char trans, CoefficientPrecision alpha, const HMatrix<CoefficientPrecision, CoordinatePrecision> &A, const CoefficientPrecision *in, CoefficientPrecision beta, CoefficientPrecision *out) {
    std::vector<const HMatrix<CoefficientPrecision, CoordinatePrecision> *> leaves;
    std::vector<const HMatrix<CoefficientPrecision, CoordinatePrecision> *> leaves_for_symmetry;
    std::tie(leaves, leaves_for_symmetry) = get_leaves_from(A); // C++17 structured binding

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
            internal_add_hmatrix_vector_product(trans, CoefficientPrecision(1), *leaves[b], in + input_offset - local_input_offset, CoefficientPrecision(1), temp.data() + (output_offset - local_output_offset));
        }

        // Symmetry part of the diagonal part
        if (A.get_symmetry_for_leaves() != 'N') {
#if defined(_OPENMP)
#    pragma omp for schedule(guided) nowait
#endif
            for (int b = 0; b < leaves_for_symmetry.size(); b++) {
                int input_offset  = (leaves_for_symmetry[b]->*get_input_cluster)().get_offset();
                int output_offset = (leaves_for_symmetry[b]->*get_output_cluster)().get_offset();
                internal_add_hmatrix_vector_product(trans_sym, CoefficientPrecision(1), *leaves_for_symmetry[b], in + output_offset - local_input_offset, CoefficientPrecision(1), temp.data() + (input_offset - local_output_offset));
            }
        }

#if defined(_OPENMP)
#    pragma omp critical
#endif
        Blas<CoefficientPrecision>::axpy(&out_size, &alpha, temp.data(), &incx, out, &incy);
    }
}

template <typename ExecutionPolicy, typename CoefficientPrecision, typename CoordinatePrecision = underlying_type<CoefficientPrecision>>
void add_hmatrix_vector_product(ExecutionPolicy &&, char trans, CoefficientPrecision alpha, const HMatrix<CoefficientPrecision, CoordinatePrecision> &A, const CoefficientPrecision *in, CoefficientPrecision beta, CoefficientPrecision *out, CoefficientPrecision *buffer = nullptr) {
    auto &source_cluster = A.get_source_cluster();
    auto &target_cluster = A.get_target_cluster();
    std::vector<CoefficientPrecision> tmp(buffer == nullptr ? target_cluster.get_size() + source_cluster.get_size() : 0, 0);
    CoefficientPrecision *buffer_ptr = buffer == nullptr ? tmp.data() : buffer;
    user_to_cluster(source_cluster, in, buffer_ptr);
    user_to_cluster(target_cluster, out, buffer_ptr + source_cluster.get_size());

#if defined(__cpp_lib_execution) && __cplusplus >= 201703L
    if constexpr (std::is_execution_policy_v<std::decay_t<ExecutionPolicy>>) {
        if constexpr (std::is_same_v<std::decay_t<ExecutionPolicy>, std::execution::parallel_policy>) {
            openmp_internal_add_hmatrix_vector_product(trans, alpha, A, buffer_ptr, beta, buffer_ptr + source_cluster.get_size());
        } else if constexpr (std::is_same_v<std::decay_t<ExecutionPolicy>, std::execution::sequenced_policy>) {
            sequential_internal_add_hmatrix_vector_product(trans, alpha, A, buffer_ptr, beta, buffer_ptr + source_cluster.get_size());
        } else {
            static_assert(std::is_same_v<std::decay_t<ExecutionPolicy>, std::execution::sequenced_policy> || std::is_same_v<std::decay_t<ExecutionPolicy>, std::execution::parallel_policy>, "Invalid execution policy for add_hmatrix_vector_product.");
        }
    } else {
        static_assert(std::is_execution_policy_v<std::decay_t<ExecutionPolicy>>, "Invalid execution policy for add_hmatrix_vector_product.");
    }
#else
    openmp_internal_add_hmatrix_vector_product(trans, alpha, A, buffer_ptr, beta, buffer_ptr + source_cluster.get_size());
#endif
    cluster_to_user(target_cluster, buffer_ptr + source_cluster.get_size(), out);
}

template <typename CoefficientPrecision, typename CoordinatePrecision = underlying_type<CoefficientPrecision>>
void add_hmatrix_vector_product(char trans, CoefficientPrecision alpha, const HMatrix<CoefficientPrecision, CoordinatePrecision> &A, const CoefficientPrecision *in, CoefficientPrecision beta, CoefficientPrecision *out, CoefficientPrecision *buffer = nullptr) {
#if defined(__cpp_lib_execution) && __cplusplus >= 201703L
    add_hmatrix_vector_product(std::execution::par, trans, alpha, A, in, beta, out, buffer);
#else
    add_hmatrix_vector_product(nullptr, trans, alpha, A, in, beta, out, buffer);
#endif
}

} // namespace htool

#endif
