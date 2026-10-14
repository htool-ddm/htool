#ifndef HTOOL_HMATRIX_LINALG_ADD_HMATRIX_VECTOR_PRODUCT_HPP
#define HTOOL_HMATRIX_LINALG_ADD_HMATRIX_VECTOR_PRODUCT_HPP

#include "../../clustering/cluster_node.hpp"
#include "../../misc/misc.hpp"
#include "../execution_policies.hpp"
#include "../hmatrix.hpp"
#include "omp/add_hmatrix_vector_product.hpp"
#include "seq/add_hmatrix_vector_product.hpp"
#include <vector>

namespace htool {

template <typename ExecutionPolicy, typename CoefficientPrecision, typename CoordinatePrecision = underlying_type<CoefficientPrecision>>
void add_hmatrix_vector_product(ExecutionPolicy &&, char trans, CoefficientPrecision alpha, const HMatrix<CoefficientPrecision, CoordinatePrecision> &A, const CoefficientPrecision *in, CoefficientPrecision beta, CoefficientPrecision *out, CoefficientPrecision *buffer = nullptr) {
    auto &source_cluster = A.get_source_cluster();
    auto &target_cluster = A.get_target_cluster();
    std::vector<CoefficientPrecision> tmp(buffer == nullptr ? target_cluster.get_size() + source_cluster.get_size() : 0, 0);
    CoefficientPrecision *buffer_ptr = buffer == nullptr ? tmp.data() : buffer;
    user_to_cluster(source_cluster, in, buffer_ptr);
    user_to_cluster(target_cluster, out, buffer_ptr + source_cluster.get_size());

#if __cplusplus >= 201703L
    if constexpr (is_execution_policy_v<std::decay_t<ExecutionPolicy>>) {
        if constexpr (std::is_same_v<std::decay_t<ExecutionPolicy>, exec_compat::parallel_policy>) {
            openmp_internal_add_hmatrix_vector_product(trans, alpha, A, buffer_ptr, beta, buffer_ptr + source_cluster.get_size());
        } else if constexpr (std::is_same_v<std::decay_t<ExecutionPolicy>, exec_compat::sequenced_policy>) {
            sequential_internal_add_hmatrix_vector_product(trans, alpha, A, buffer_ptr, beta, buffer_ptr + source_cluster.get_size());
        } else {
            static_assert(std::is_same_v<std::decay_t<ExecutionPolicy>, exec_compat::sequenced_policy> || std::is_same_v<std::decay_t<ExecutionPolicy>, exec_compat::parallel_policy>, "Invalid execution policy for add_hmatrix_vector_product.");
        }
    } else {
        static_assert(is_execution_policy_v<std::decay_t<ExecutionPolicy>>, "Invalid execution policy for add_hmatrix_vector_product.");
    }
#else
    sequential_internal_add_hmatrix_vector_product(trans, alpha, A, buffer_ptr, beta, buffer_ptr + source_cluster.get_size());
#endif
    cluster_to_user(target_cluster, buffer_ptr + source_cluster.get_size(), out);
}

template <typename CoefficientPrecision, typename CoordinatePrecision = underlying_type<CoefficientPrecision>>
void add_hmatrix_vector_product(char trans, CoefficientPrecision alpha, const HMatrix<CoefficientPrecision, CoordinatePrecision> &A, const CoefficientPrecision *in, CoefficientPrecision beta, CoefficientPrecision *out, CoefficientPrecision *buffer = nullptr) {
#if __cplusplus >= 201703L
    add_hmatrix_vector_product(exec_compat::seq, trans, alpha, A, in, beta, out, buffer);
#else
    add_hmatrix_vector_product(nullptr, trans, alpha, A, in, beta, out, buffer);
#endif
}

} // namespace htool

#endif
