#ifndef HTOOL_HMATRIX_LINALG_ADD_HMATRIX_MATRIX_PRODUCT_HPP
#define HTOOL_HMATRIX_LINALG_ADD_HMATRIX_MATRIX_PRODUCT_HPP

#include "../../clustering/cluster_node.hpp"
#include "../../matrix/matrix.hpp"
#include "../../misc/misc.hpp"
#include "../execution_policies.hpp"
#include "../hmatrix.hpp"
#include "omp/add_hmatrix_matrix_product.hpp"
#include "omp_task/add_hmatrix_matrix_product.hpp"
#include "seq/add_hmatrix_matrix_product.hpp"
#include <vector>

namespace htool {

template <typename ExecutionPolicy,
          typename MatB,
          typename MatC,
          typename CoordinatePrecision = underlying_type<typename MatB::value_type>,
          typename                     = std::enable_if_t<
                                  std::is_same<typename MatB::value_type, typename MatC::value_type>::value>>
void internal_add_hmatrix_matrix_product(ExecutionPolicy &&execution_policy, char transa, char transb, typename MatB::value_type alpha, const HMatrix<typename MatB::value_type, CoordinatePrecision> &A, const MatB &B, typename MatB::value_type beta, MatC &C) {
    (void)execution_policy; // trick to avoid triggering unused parameter. C++17 -> use [[maybe_unused]] attribute
#if __cplusplus >= 201703L
    if constexpr (is_execution_policy_v<std::decay_t<ExecutionPolicy>>) {
        if constexpr (std::is_same_v<std::decay_t<ExecutionPolicy>, exec_compat::parallel_policy>) {
            openmp_internal_add_hmatrix_matrix_product(transa, transb, alpha, A, B, beta, C);
        } else if constexpr (std::is_same_v<std::decay_t<ExecutionPolicy>, exec_compat::sequenced_policy>) {
            internal_add_hmatrix_matrix_product(transa, transb, alpha, A, B, beta, C);
        } else if constexpr (std::is_same_v<std::decay_t<ExecutionPolicy>, omp_task_policy<typename MatB::value_type, CoordinatePrecision>>) {
            auto &hmatrix_task_dependencies = execution_policy.hmatrix_task_dependencies;
            update_L0(hmatrix_task_dependencies, A);
            const auto &in_L0  = transa == 'N' ? hmatrix_task_dependencies.source_L0 : hmatrix_task_dependencies.target_L0;
            const auto &out_L0 = transa == 'N' ? hmatrix_task_dependencies.target_L0 : hmatrix_task_dependencies.source_L0;
            run_task_based(
                hmatrix_task_dependencies, A, [&]() { internal_add_hmatrix_matrix_product(transa, transb, alpha, A, B, beta, C); }, [&](auto &L0) { task_based_internal_add_hmatrix_matrix_product(transa, transb, alpha, A, B, beta, C, L0, in_L0, out_L0); });
        } else {
            static_assert(std::is_same_v<std::decay_t<ExecutionPolicy>, exec_compat::sequenced_policy> || std::is_same_v<std::decay_t<ExecutionPolicy>, exec_compat::parallel_policy> || std::is_same_v<std::decay_t<ExecutionPolicy>, omp_task_policy<typename MatB::value_type, CoordinatePrecision>>, "Invalid execution policy for add_hmatrix_matrix_product.");
        }
    } else {
        static_assert(is_execution_policy_v<std::decay_t<ExecutionPolicy>>, "Invalid execution policy for add_hmatrix_matrix_product.");
    }
#else
    internal_add_hmatrix_matrix_product(transa, transb, alpha, A, B, beta, C);
#endif
}

template <typename ExecutionPolicy,
          typename MatB,
          typename MatC,
          typename CoordinatePrecision = underlying_type<typename MatB::value_type>,
          typename                     = std::enable_if_t<
                                  std::is_same<typename MatB::value_type, typename MatC::value_type>::value>>
void add_hmatrix_matrix_product(ExecutionPolicy &&execution_policy, char transa, char transb, typename MatB::value_type alpha, const HMatrix<typename MatB::value_type, CoordinatePrecision> &A, const MatB &B, typename MatB::value_type beta, MatC &C, typename MatB::value_type *buffer = nullptr) {
    (void)execution_policy; // trick to avoid triggering unused parameter. C++17 -> use [[maybe_unused]] attribute

    using CoefficientPrecision = typename MatB::value_type;
    auto &target_cluster       = A.get_target_cluster();
    auto &source_cluster       = A.get_source_cluster();

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

#if __cplusplus >= 201703L
    internal_add_hmatrix_matrix_product(execution_policy, transa, transb, alpha, A, permuted_B, beta, permuted_C);
#else
    internal_add_hmatrix_matrix_product(transa, transb, alpha, A, permuted_B, beta, permuted_C);
#endif
    for (int i = 0; i < C.nb_cols(); i++) {
        cluster_to_user(target_cluster, permuted_C.data() + target_cluster.get_size() * i, C.data() + target_cluster.get_size() * i);
    }
}

template <typename MatB,
          typename MatC,
          typename CoordinatePrecision = underlying_type<typename MatB::value_type>,
          typename                     = std::enable_if_t<
                                  std::is_same<typename MatB::value_type, typename MatC::value_type>::value>>
void add_hmatrix_matrix_product(char transa, char transb, typename MatB::value_type alpha, const HMatrix<typename MatB::value_type, CoordinatePrecision> &A, const MatB &B, typename MatB::value_type beta, MatC &C, typename MatB::value_type *buffer = nullptr) {
#if __cplusplus >= 201703L
    add_hmatrix_matrix_product(exec_compat::seq, transa, transb, alpha, A, B, beta, C, buffer);
#else
    add_hmatrix_matrix_product(nullptr, transa, transb, alpha, A, B, beta, C, buffer);
#endif
}

} // namespace htool

#endif
