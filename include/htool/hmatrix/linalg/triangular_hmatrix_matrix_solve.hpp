#ifndef HTOOL_HMATRIX_LINALG_TRIANGULAR_HMATRIX_MATRIX_SOLVE_HPP
#define HTOOL_HMATRIX_LINALG_TRIANGULAR_HMATRIX_MATRIX_SOLVE_HPP

#include "../../matrix/linalg/scale.hpp"
#include "../../matrix/matrix.hpp"
#include "../../misc/misc.hpp"
#include "../execution_policies.hpp"
#include "../hmatrix.hpp"
#include "omp_task/apply_ldlt_diagonal.hpp"
#include "omp_task/triangular_hmatrix_matrix_solve.hpp"
#include "seq/apply_ldlt_diagonal.hpp"
#include "seq/triangular_hmatrix_matrix_solve.hpp"
#include "seq/triangular_ldlt_hmatrix_matrix_solve.hpp"
#include <vector>

// The single triangular solves and the application of D^-1 of the solves in solve.hpp, used by LocalHMatrixPlusOverlapSolver.

namespace htool {

template <typename ExecutionPolicy, typename CoefficientPrecision, typename CoordinatePrecision>
void internal_triangular_hmatrix_matrix_solve(ExecutionPolicy &&execution_policy, char side, char UPLO, char transa, char diag, CoefficientPrecision alpha, const HMatrix<CoefficientPrecision, CoordinatePrecision> &A, Matrix<CoefficientPrecision> &B) {
    (void)execution_policy; // trick to avoid triggering unused parameter. C++17 -> use [[maybe_unused]] attribute

#if __cplusplus >= 201703L
    if constexpr (is_execution_policy_v<std::decay_t<ExecutionPolicy>>) {
        if constexpr (std::is_same_v<std::decay_t<ExecutionPolicy>, exec_compat::parallel_policy>) {
            HMatrixTaskDependencies<CoefficientPrecision, CoordinatePrecision> hmatrix_task_dependencies;
            hmatrix_task_dependencies.set_L0(A);
            if (alpha != CoefficientPrecision(1)) {
                scale(alpha, B);
            }
#    if defined(_OPENMP)
#        pragma omp parallel
#        pragma omp single
#    endif
            {
                task_based_internal_triangular_hmatrix_matrix_solve(side, UPLO, transa, diag, A, B, hmatrix_task_dependencies.L0);
            }
        } else if constexpr (std::is_same_v<std::decay_t<ExecutionPolicy>, exec_compat::sequenced_policy>) {
            internal_triangular_hmatrix_matrix_solve(side, UPLO, transa, diag, alpha, A, B);
        } else if constexpr (std::is_same_v<std::decay_t<ExecutionPolicy>, omp_task_policy<CoefficientPrecision, CoordinatePrecision>>) {
            update_L0(execution_policy.hmatrix_task_dependencies, A);
            if (alpha != CoefficientPrecision(1)) {
                scale(alpha, B);
            }
            run_task_based(
                execution_policy.hmatrix_task_dependencies, A, [&]() { internal_triangular_hmatrix_matrix_solve(side, UPLO, transa, diag, CoefficientPrecision(1), A, B); }, [&](auto &L0) { task_based_internal_triangular_hmatrix_matrix_solve(side, UPLO, transa, diag, A, B, L0); });
        } else {
            static_assert(std::is_same_v<std::decay_t<ExecutionPolicy>, exec_compat::sequenced_policy> || std::is_same_v<std::decay_t<ExecutionPolicy>, exec_compat::parallel_policy> || std::is_same_v<std::decay_t<ExecutionPolicy>, omp_task_policy<CoefficientPrecision, CoordinatePrecision>>, "Invalid execution policy for solve.");
        }
    } else {
        static_assert(is_execution_policy_v<std::decay_t<ExecutionPolicy>>, "Invalid execution policy for solve.");
    }
#else
    internal_triangular_hmatrix_matrix_solve(side, UPLO, transa, diag, alpha, A, B);
#endif
}

template <typename ExecutionPolicy, typename CoefficientPrecision, typename CoordinatePrecision>
void internal_triangular_ldlt_hmatrix_matrix_solve(ExecutionPolicy &&execution_policy, char side, char UPLO, char transa, CoefficientPrecision alpha, const HMatrix<CoefficientPrecision, CoordinatePrecision> &A, Matrix<CoefficientPrecision> &B) {
    (void)execution_policy; // trick to avoid triggering unused parameter. C++17 -> use [[maybe_unused]] attribute

#if __cplusplus >= 201703L
    if constexpr (is_execution_policy_v<std::decay_t<ExecutionPolicy>>) {
        if constexpr (std::is_same_v<std::decay_t<ExecutionPolicy>, exec_compat::parallel_policy>) {
            HMatrixTaskDependencies<CoefficientPrecision, CoordinatePrecision> hmatrix_task_dependencies;
            hmatrix_task_dependencies.set_L0(A);
            if (alpha != CoefficientPrecision(1)) {
                scale(alpha, B);
            }
#    if defined(_OPENMP)
#        pragma omp parallel
#        pragma omp single
#    endif
            {
                task_based_internal_triangular_ldlt_hmatrix_matrix_solve(side, UPLO, transa, A, B, hmatrix_task_dependencies.L0);
            }
        } else if constexpr (std::is_same_v<std::decay_t<ExecutionPolicy>, exec_compat::sequenced_policy>) {
            internal_triangular_ldlt_hmatrix_matrix_solve(side, UPLO, transa, alpha, A, B);
        } else if constexpr (std::is_same_v<std::decay_t<ExecutionPolicy>, omp_task_policy<CoefficientPrecision, CoordinatePrecision>>) {
            update_L0(execution_policy.hmatrix_task_dependencies, A);
            if (alpha != CoefficientPrecision(1)) {
                scale(alpha, B);
            }
            run_task_based(
                execution_policy.hmatrix_task_dependencies, A, [&]() { internal_triangular_ldlt_hmatrix_matrix_solve(side, UPLO, transa, CoefficientPrecision(1), A, B); }, [&](auto &L0) { task_based_internal_triangular_ldlt_hmatrix_matrix_solve(side, UPLO, transa, A, B, L0); });
        } else {
            static_assert(std::is_same_v<std::decay_t<ExecutionPolicy>, exec_compat::sequenced_policy> || std::is_same_v<std::decay_t<ExecutionPolicy>, exec_compat::parallel_policy> || std::is_same_v<std::decay_t<ExecutionPolicy>, omp_task_policy<CoefficientPrecision, CoordinatePrecision>>, "Invalid execution policy for solve.");
        }
    } else {
        static_assert(is_execution_policy_v<std::decay_t<ExecutionPolicy>>, "Invalid execution policy for solve.");
    }
#else
    internal_triangular_ldlt_hmatrix_matrix_solve(side, UPLO, transa, alpha, A, B);
#endif
}

template <typename ExecutionPolicy, typename CoefficientPrecision, typename CoordinatePrecision>
void internal_apply_ldlt_diagonal(ExecutionPolicy &&execution_policy, char symmetry, char side, char UPLO, const HMatrix<CoefficientPrecision, CoordinatePrecision> &A, Matrix<CoefficientPrecision> &B, int offset) {
    (void)execution_policy; // trick to avoid triggering unused parameter. C++17 -> use [[maybe_unused]] attribute

#if __cplusplus >= 201703L
    if constexpr (is_execution_policy_v<std::decay_t<ExecutionPolicy>>) {
        if constexpr (std::is_same_v<std::decay_t<ExecutionPolicy>, exec_compat::parallel_policy>) {
            HMatrixTaskDependencies<CoefficientPrecision, CoordinatePrecision> hmatrix_task_dependencies;
            hmatrix_task_dependencies.set_L0(A);
#    if defined(_OPENMP)
#        pragma omp parallel
#        pragma omp single
#    endif
            {
                task_based_internal_apply_ldlt_diagonal(symmetry, side, UPLO, B, offset, hmatrix_task_dependencies.L0);
            }
        } else if constexpr (std::is_same_v<std::decay_t<ExecutionPolicy>, exec_compat::sequenced_policy>) {
            internal_apply_ldlt_diagonal(symmetry, side, UPLO, A, B, offset);
        } else if constexpr (std::is_same_v<std::decay_t<ExecutionPolicy>, omp_task_policy<CoefficientPrecision, CoordinatePrecision>>) {
            update_L0(execution_policy.hmatrix_task_dependencies, A);
            run_task_based(
                execution_policy.hmatrix_task_dependencies, A, [&]() { internal_apply_ldlt_diagonal(symmetry, side, UPLO, A, B, offset); }, [&](auto &L0) { task_based_internal_apply_ldlt_diagonal(symmetry, side, UPLO, B, offset, L0); });
        } else {
            static_assert(std::is_same_v<std::decay_t<ExecutionPolicy>, exec_compat::sequenced_policy> || std::is_same_v<std::decay_t<ExecutionPolicy>, exec_compat::parallel_policy> || std::is_same_v<std::decay_t<ExecutionPolicy>, omp_task_policy<CoefficientPrecision, CoordinatePrecision>>, "Invalid execution policy for solve.");
        }
    } else {
        static_assert(is_execution_policy_v<std::decay_t<ExecutionPolicy>>, "Invalid execution policy for solve.");
    }
#else
    internal_apply_ldlt_diagonal(symmetry, side, UPLO, A, B, offset);
#endif
}

} // namespace htool

#endif
