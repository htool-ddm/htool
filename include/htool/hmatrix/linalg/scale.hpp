#ifndef HTOOL_HMATRIX_LINALG_SCALE_HPP
#define HTOOL_HMATRIX_LINALG_SCALE_HPP

#include "../execution_policies.hpp"
#include "../hmatrix.hpp"
#include "omp/scale.hpp"
#include "seq/scale.hpp"

namespace htool {

template <typename ExecutionPolicy, typename CoefficientPrecision, typename CoordinatePrecision = underlying_type<CoefficientPrecision>>
void scale(ExecutionPolicy &&, CoefficientPrecision da, HMatrix<CoefficientPrecision, CoordinatePrecision> &A) {
#if __cplusplus >= 201703L
    if constexpr (is_execution_policy_v<std::decay_t<ExecutionPolicy>>) {
        if constexpr (std::is_same_v<std::decay_t<ExecutionPolicy>, exec_compat::parallel_policy>) {
            openmp_scale(da, A);
        } else if constexpr (std::is_same_v<std::decay_t<ExecutionPolicy>, exec_compat::sequenced_policy>) {
            sequential_scale(da, A);
        } else {
            static_assert(std::is_same_v<std::decay_t<ExecutionPolicy>, exec_compat::sequenced_policy> || std::is_same_v<std::decay_t<ExecutionPolicy>, exec_compat::parallel_policy>, "Invalid execution policy for scale.");
        }
    } else {
        static_assert(is_execution_policy_v<std::decay_t<ExecutionPolicy>>, "Invalid execution policy for scale.");
    }
#else
    sequential_scale(da, A);
#endif
}

template <typename CoefficientPrecision, typename CoordinatePrecision = underlying_type<CoefficientPrecision>>
void scale(CoefficientPrecision da, HMatrix<CoefficientPrecision, CoordinatePrecision> &A) {
#if __cplusplus >= 201703L
    scale(exec_compat::seq, da, A);
#else
    scale(nullptr, da, A);
#endif
}

} // namespace htool

#endif
