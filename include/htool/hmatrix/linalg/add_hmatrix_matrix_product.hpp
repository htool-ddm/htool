#ifndef HTOOL_HMATRIX_LINALG_ADD_HMATRIX_MATRIX_PRODUCT_HPP
#define HTOOL_HMATRIX_LINALG_ADD_HMATRIX_MATRIX_PRODUCT_HPP

#include "../../matrix/linalg/scale.hpp"            // for scale
#include "../../matrix/linalg/transpose.hpp"        // for transpose
#include "../../matrix/matrix.hpp"                  // for Matrix
#include "../../matrix/utils/SVD_truncation.hpp"    // for SVD_truncation
#include "../../misc/misc.hpp"                      // for underlying_type
#include "../../wrappers/wrapper_blas.hpp"          // for Blas
#include "../hmatrix.hpp"                           // for HMatrix
#include "../lrmat/lrmat.hpp"                       // for LowRankMatrix
#include "../lrmat/utils/SVD_recompression.hpp"     // for recompression
#include "add_hmatrix_matrix_product_row_major.hpp" // for sequential_ad...
#include "execution_policies.hpp"
#include <algorithm> // for copy_n, min
#include <vector>    // for vector

namespace htool {

template <typename ExecutionPolicy, typename CoefficientPrecision, typename CoordinatePrecision = CoefficientPrecision>
void internal_add_hmatrix_matrix_product(ExecutionPolicy &&, char transa, char transb, CoefficientPrecision alpha, const HMatrix<CoefficientPrecision, CoordinatePrecision> &A, const Matrix<CoefficientPrecision> &B, CoefficientPrecision beta, Matrix<CoefficientPrecision> &C) {
    if (transb == 'N') {
        Matrix<CoefficientPrecision> transposed_B(B.nb_cols(), B.nb_rows()), transposed_C(C.nb_cols(), C.nb_rows());
        transpose(B, transposed_B);
        transpose(C, transposed_C);
#if defined(__cpp_lib_execution) && __cplusplus >= 201703L
        if constexpr (std::is_execution_policy_v<std::decay_t<ExecutionPolicy>>) {
            if constexpr (std::is_same_v<std::decay_t<ExecutionPolicy>, std::execution::parallel_policy>) {
                openmp_internal_add_hmatrix_matrix_product_row_major(transa, transb, alpha, A, transposed_B.data(), beta, transposed_C.data(), transposed_C.nb_rows());
            } else if constexpr (std::is_same_v<std::decay_t<ExecutionPolicy>, std::execution::sequenced_policy>) {
                sequential_internal_add_hmatrix_matrix_product_row_major(transa, transb, alpha, A, transposed_B.data(), beta, transposed_C.data(), transposed_C.nb_rows());
            } else {
                static_assert(false, "Invalid execution policy for add_hmatrix_vector_product.");
            }
        } else {
            static_assert(false, "Invalid execution policy for add_hmatrix_vector_product.");
        }
#else
        sequential_internal_add_hmatrix_matrix_product_row_major(transa, transb, alpha, A, transposed_B.data(), beta, transposed_C.data(), transposed_C.nb_rows());
#endif
        transpose(transposed_C, C);
    } else {
        Matrix<CoefficientPrecision> transposed_C(C.nb_cols(), C.nb_rows());
        transpose(C, transposed_C);
        std::vector<CoefficientPrecision> buffer_B(transb == 'C' ? B.nb_cols() * B.nb_rows() : 0);

        if (transb == 'C') {
            std::copy(B.data(), B.data() + B.nb_cols() * B.nb_rows(), buffer_B.data());
            conj_if_complex(buffer_B.data(), buffer_B.size());
        }

#if defined(__cpp_lib_execution) && __cplusplus >= 201703L
        if constexpr (std::is_execution_policy_v<std::decay_t<ExecutionPolicy>>) {
            if constexpr (std::is_same_v<std::decay_t<ExecutionPolicy>, std::execution::parallel_policy>) {
                openmp_internal_add_hmatrix_matrix_product_row_major(transa, 'N', alpha, A, transb == 'C' ? buffer_B.data() : B.data(), beta, transposed_C.data(), transposed_C.nb_rows());
            } else if constexpr (std::is_same_v<std::decay_t<ExecutionPolicy>, std::execution::sequenced_policy>) {
                sequential_internal_add_hmatrix_matrix_product_row_major(transa, 'N', alpha, A, transb == 'C' ? buffer_B.data() : B.data(), beta, transposed_C.data(), transposed_C.nb_rows());
            } else {
                static_assert(false, "Invalid execution policy for add_hmatrix_vector_product.");
            }
        } else {
            static_assert(false, "Invalid execution policy for add_hmatrix_vector_product.");
        }
#else
        sequential_internal_add_hmatrix_matrix_product_row_major(transa, 'N', alpha, A, transb == 'C' ? buffer_B.data() : B.data(), beta, transposed_C.data(), transposed_C.nb_rows());
#endif
        transpose(transposed_C, C);
    }
}

template <typename CoefficientPrecision, typename CoordinatePrecision = CoefficientPrecision>
void internal_add_hmatrix_matrix_product(char transa, char transb, CoefficientPrecision alpha, const HMatrix<CoefficientPrecision, CoordinatePrecision> &A, const Matrix<CoefficientPrecision> &B, CoefficientPrecision beta, Matrix<CoefficientPrecision> &C) {
#if defined(__cpp_lib_execution) && __cplusplus >= 201703L
    internal_add_hmatrix_matrix_product(std::execution::seq, transa, transb, alpha, A, B, beta, C);
#else
    internal_add_hmatrix_matrix_product(nullptr, transa, transb, alpha, A, B, beta, C);
#endif
}

template <typename CoefficientPrecision, typename CoordinatePrecision = CoefficientPrecision>
void internal_add_hmatrix_matrix_product(char transa, char transb, CoefficientPrecision alpha, const HMatrix<CoefficientPrecision, CoordinatePrecision> &A, const Matrix<CoefficientPrecision> &B, CoefficientPrecision beta, LowRankMatrix<CoefficientPrecision> &C) {
    bool C_is_overwritten = (beta == CoefficientPrecision(0) || C.rank_of() == 0);

    int nb_rows = (transa == 'N') ? A.nb_rows() : A.nb_cols();
    int nb_cols = (transb == 'N') ? B.nb_cols() : B.nb_rows();

    //
    Matrix<CoefficientPrecision> AB(nb_rows, nb_cols);
    internal_add_hmatrix_matrix_product(transa, transb, alpha, A, B, CoefficientPrecision(0), AB);

    // SVD truncation
    std::vector<underlying_type<CoefficientPrecision>> singular_values(std::min(nb_rows, nb_cols));
    Matrix<CoefficientPrecision> u(nb_rows, nb_rows);
    Matrix<CoefficientPrecision> vt(nb_cols, nb_cols);
    int truncated_rank = SVD_truncation(AB, C.get_epsilon(), u, vt, singular_values);

    // new_U=u*sqrt(tildeS) and new_V=sqrt(tildeS)*vt in the right dimensions
    Matrix<CoefficientPrecision> *new_U_ptr, *new_V_ptr;
    Matrix<CoefficientPrecision> U_1, V_1;
    if (C_is_overwritten) {
        new_U_ptr = &C.get_U();
        new_V_ptr = &C.get_V();
    } else {
        new_U_ptr = &U_1;
        new_V_ptr = &V_1;
    }

    {
        Matrix<CoefficientPrecision> &new_U = *new_U_ptr;
        Matrix<CoefficientPrecision> &new_V = *new_V_ptr;
        int M                               = nb_rows;
        int N                               = nb_cols;
        int incx                            = 1;
        new_U.resize(M, truncated_rank);
        new_V.resize(truncated_rank, N);
        CoefficientPrecision scaling_coef;
        for (int r = 0; r < truncated_rank; r++) {
            scaling_coef = std::sqrt(singular_values[r]);
            std::copy_n(u.data() + r * u.nb_rows(), u.nb_cols(), new_U.data() + r * M);
            Blas<CoefficientPrecision>::scal(&M, &scaling_coef, new_U.data() + r * M, &incx);
        }
        for (int r = 0; r < vt.nb_cols(); r++) {
            std::copy_n(vt.data() + r * vt.nb_rows(), truncated_rank, new_V.data() + r * truncated_rank);
        }

        for (int r = 0; r < truncated_rank; r++) {
            for (int j = 0; j < new_V.nb_cols(); j++) {
                new_V(r, j) = std::sqrt(singular_values[r]) * new_V(r, j);
            }
        }
    }

    if (C_is_overwritten) {
        return;
    }

    // Concatenate U_1 and U_2
    Matrix<CoefficientPrecision> &U_2 = C.get_U();
    Matrix<CoefficientPrecision> new_U(U_1.nb_rows(), U_1.nb_cols() + U_2.nb_cols());
    std::copy_n(U_1.data(), U_1.nb_rows() * U_1.nb_cols(), new_U.data());
    std::copy_n(U_2.data(), U_2.nb_rows() * U_2.nb_cols(), new_U.data() + U_1.nb_rows() * U_1.nb_cols());

    // Concatenate V_1 and V_2
    Matrix<CoefficientPrecision> &V_2 = C.get_V();
    scale(beta, V_2);
    Matrix<CoefficientPrecision> new_V(V_1.nb_rows() + V_2.nb_rows(), V_2.nb_cols());
    for (int j = 0; j < new_V.nb_cols(); j++) {
        std::copy_n(V_1.data() + j * V_1.nb_rows(), V_1.nb_rows(), new_V.data() + j * new_V.nb_rows());
        std::copy_n(V_2.data() + j * V_2.nb_rows(), V_2.nb_rows(), new_V.data() + j * new_V.nb_rows() + V_1.nb_rows());
    }

    // Set C
    C.get_U() = new_U;
    C.get_V() = new_V;
    SVD_recompression(C);
}

template <typename ExecutionPolicy, typename CoefficientPrecision, typename CoordinatePrecision = CoefficientPrecision>
void add_hmatrix_matrix_product(ExecutionPolicy &&execution_policy, char transa, char transb, CoefficientPrecision alpha, const HMatrix<CoefficientPrecision, CoordinatePrecision> &A, const Matrix<CoefficientPrecision> &B, CoefficientPrecision beta, Matrix<CoefficientPrecision> &C, CoefficientPrecision *buffer = nullptr) {
    auto &target_cluster = A.get_target_cluster();
    auto &source_cluster = A.get_source_cluster();

    std::vector<CoefficientPrecision> tmp(buffer == nullptr ? (target_cluster.get_size() + source_cluster.get_size()) * B.nb_cols() : 0, 0);

    Matrix<CoefficientPrecision> permuted_B;
    Matrix<CoefficientPrecision> permuted_C;
    permuted_B.assign(B.nb_rows(), B.nb_cols(), (buffer == nullptr) ? tmp.data() : buffer, false);
    permuted_C.assign(C.nb_rows(), C.nb_cols(), (buffer == nullptr) ? tmp.data() + B.nb_rows() * B.nb_cols() : buffer + B.nb_rows() * B.nb_cols(), false);

    for (int i = 0; i < B.nb_cols(); i++) {
        user_to_cluster(source_cluster, B.data() + source_cluster.get_size() * i, permuted_B.data() + source_cluster.get_size() * i);
    }
    for (int i = 0; i < C.nb_cols(); i++) {
        user_to_cluster(target_cluster, C.data() + target_cluster.get_size() * i, permuted_C.data() + target_cluster.get_size() * i);
    }

#if defined(__cpp_lib_execution) && __cplusplus >= 201703L
    internal_add_hmatrix_matrix_product(execution_policy, transa, transb, alpha, A, permuted_B, beta, permuted_C);
#else
    internal_add_hmatrix_matrix_product(transa, transb, alpha, A, permuted_B, beta, permuted_C);
#endif
    for (int i = 0; i < C.nb_cols(); i++) {
        cluster_to_user(target_cluster, permuted_C.data() + target_cluster.get_size() * i, C.data() + target_cluster.get_size() * i);
    }
}

template <typename CoefficientPrecision, typename CoordinatePrecision = CoefficientPrecision>
void add_hmatrix_matrix_product(char transa, char transb, CoefficientPrecision alpha, const HMatrix<CoefficientPrecision, CoordinatePrecision> &A, const Matrix<CoefficientPrecision> &B, CoefficientPrecision beta, Matrix<CoefficientPrecision> &C, CoefficientPrecision *buffer = nullptr) {
#if defined(__cpp_lib_execution) && __cplusplus >= 201703L
    add_hmatrix_matrix_product(std::execution::seq, transa, transb, alpha, A, B, beta, C, buffer);
#else
    add_hmatrix_matrix_product(nullptr, transa, transb, alpha, A, B, beta, C, buffer);
#endif
}
} // namespace htool

#endif
