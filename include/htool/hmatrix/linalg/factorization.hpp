#ifndef HTOOL_HMATRIX_LINALG_FACTORIZATION_HPP
#define HTOOL_HMATRIX_LINALG_FACTORIZATION_HPP

#include "../../misc/logger.hpp"
#include "../../misc/misc.hpp"
#include "../execution_policies.hpp"
#include "../hmatrix.hpp"
#include "omp_task/factorization.hpp"
#include "seq/factorization.hpp"
#include "solve.hpp"
#include <vector>

namespace htool {

template <typename ExecutionPolicy, typename CoefficientPrecision, typename CoordinatePrecision = underlying_type<CoefficientPrecision>>
void lu_factorization(ExecutionPolicy &&execution_policy, HMatrix<CoefficientPrecision, CoordinatePrecision> &hmatrix) {
    if (!hmatrix.is_block_tree_consistent()) {
        htool::Logger::get_instance().log(LogLevel::ERROR, "lu_factorization is only implemented for consistent block tree."); // LCOV_EXCL_LINE
    }
    if (hmatrix.get_UPLO() != 'N') {
        htool::Logger::get_instance().log(LogLevel::ERROR, "lu_factorization cannot be used on a HMatrix with UPLO=" + std::string(1, hmatrix.get_UPLO()) + "!=N. You should use another factorization."); // LCOV_EXCL_LINE
    }

#if __cplusplus >= 201703L
    if constexpr (is_execution_policy_v<std::decay_t<ExecutionPolicy>>) {
        if constexpr (std::is_same_v<std::decay_t<ExecutionPolicy>, exec_compat::parallel_policy>) {
            HMatrixTaskDependencies<CoefficientPrecision, CoordinatePrecision> hmatrix_task_dependencies;
            hmatrix_task_dependencies.set_L0(hmatrix);
#    if defined(_OPENMP)
#        pragma omp parallel
#        pragma omp single
#    endif
            {
                task_based_lu_factorization(hmatrix, hmatrix_task_dependencies.L0);
            }
        } else if constexpr (std::is_same_v<std::decay_t<ExecutionPolicy>, exec_compat::sequenced_policy>) {
            sequential_lu_factorization(hmatrix);
        } else if constexpr (std::is_same_v<std::decay_t<ExecutionPolicy>, omp_task_policy<CoefficientPrecision, CoordinatePrecision>>) {
            update_L0(execution_policy.hmatrix_task_dependencies, hmatrix);
            run_task_based(
                execution_policy.hmatrix_task_dependencies, hmatrix, [&]() { sequential_lu_factorization(hmatrix); }, [&](auto &L0) { task_based_lu_factorization(hmatrix, L0); });
        } else {
            static_assert(std::is_same_v<std::decay_t<ExecutionPolicy>, exec_compat::sequenced_policy> || std::is_same_v<std::decay_t<ExecutionPolicy>, exec_compat::parallel_policy> || std::is_same_v<std::decay_t<ExecutionPolicy>, omp_task_policy<CoefficientPrecision, CoordinatePrecision>>, "Invalid execution policy for factorization.");
        }
    } else {
        static_assert(is_execution_policy_v<std::decay_t<ExecutionPolicy>>, "Invalid execution policy for factorization.");
    }
#else
    (void)execution_policy;
    sequential_lu_factorization(hmatrix);
#endif
}

template <typename CoefficientPrecision, typename CoordinatePrecision = underlying_type<CoefficientPrecision>>
void lu_factorization(HMatrix<CoefficientPrecision, CoordinatePrecision> &hmatrix) {
#if __cplusplus >= 201703L
    lu_factorization(exec_compat::seq, hmatrix);
#else
    lu_factorization(nullptr, hmatrix);
#endif
}

template <typename ExecutionPolicy, typename CoefficientPrecision, typename CoordinatePrecision = underlying_type<CoefficientPrecision>>
void cholesky_factorization(ExecutionPolicy &&execution_policy, char UPLO, HMatrix<CoefficientPrecision, CoordinatePrecision> &hmatrix) {
    if (!hmatrix.is_block_tree_consistent()) {
        htool::Logger::get_instance().log(LogLevel::ERROR, "cholesky_factorization is only implemented for consistent block tree."); // LCOV_EXCL_LINE
    }
    if (UPLO != 'L' and UPLO != 'U') {
        htool::Logger::get_instance().log(LogLevel::ERROR, "cholesky_factorization: UPLO=" + std::string(1, UPLO) + ", expected L or U."); // LCOV_EXCL_LINE
    }
    // A full-storage (symmetry='N') hmatrix is accepted, the caller vouching for its symmetry.
    if (hmatrix.get_symmetry() == 'S' and is_complex<CoefficientPrecision>()) {
        htool::Logger::get_instance().log(LogLevel::ERROR, "cholesky_factorization cannot be used on a complex symmetric (non-Hermitian) HMatrix. You should use ldlt_factorization."); // LCOV_EXCL_LINE
    }

#if __cplusplus >= 201703L
    if constexpr (is_execution_policy_v<std::decay_t<ExecutionPolicy>>) {
        if constexpr (std::is_same_v<std::decay_t<ExecutionPolicy>, exec_compat::parallel_policy>) {
            HMatrixTaskDependencies<CoefficientPrecision, CoordinatePrecision> hmatrix_task_dependencies;
            hmatrix_task_dependencies.set_L0(hmatrix);

#    if defined(_OPENMP)
#        pragma omp parallel
#        pragma omp single
#    endif
            {
                task_based_cholesky_factorization(UPLO, hmatrix, hmatrix_task_dependencies.L0);
            }
        } else if constexpr (std::is_same_v<std::decay_t<ExecutionPolicy>, exec_compat::sequenced_policy>) {
            sequential_cholesky_factorization(UPLO, hmatrix);
        } else if constexpr (std::is_same_v<std::decay_t<ExecutionPolicy>, omp_task_policy<CoefficientPrecision, CoordinatePrecision>>) {
            update_L0(execution_policy.hmatrix_task_dependencies, hmatrix);
            run_task_based(
                execution_policy.hmatrix_task_dependencies, hmatrix, [&]() { sequential_cholesky_factorization(UPLO, hmatrix); }, [&](auto &L0) { task_based_cholesky_factorization(UPLO, hmatrix, L0); });
        } else {
            static_assert(std::is_same_v<std::decay_t<ExecutionPolicy>, exec_compat::sequenced_policy> || std::is_same_v<std::decay_t<ExecutionPolicy>, exec_compat::parallel_policy> || std::is_same_v<std::decay_t<ExecutionPolicy>, omp_task_policy<CoefficientPrecision, CoordinatePrecision>>, "Invalid execution policy for factorization.");
        }
    } else {
        static_assert(is_execution_policy_v<std::decay_t<ExecutionPolicy>>, "Invalid execution policy for factorization.");
    }
#else
    (void)execution_policy;
    sequential_cholesky_factorization(UPLO, hmatrix);
#endif
}

template <typename CoefficientPrecision, typename CoordinatePrecision = underlying_type<CoefficientPrecision>>
void cholesky_factorization(char UPLO, HMatrix<CoefficientPrecision, CoordinatePrecision> &hmatrix) {
#if __cplusplus >= 201703L
    cholesky_factorization(exec_compat::seq, UPLO, hmatrix);
#else
    cholesky_factorization(nullptr, UPLO, hmatrix);
#endif
}

// Symmetric (symmetry='S', also complex) or Hermitian (symmetry='H') LDLt of hmatrix, stored with the
// same symmetry or in full (symmetry='N', the caller vouching for its symmetry), using its UPLO triangle.
template <typename ExecutionPolicy, typename CoefficientPrecision, typename CoordinatePrecision = underlying_type<CoefficientPrecision>>
void ldlt_factorization(ExecutionPolicy &&execution_policy, char symmetry, char UPLO, HMatrix<CoefficientPrecision, CoordinatePrecision> &hmatrix) {
    if (!hmatrix.is_block_tree_consistent()) {
        htool::Logger::get_instance().log(LogLevel::ERROR, "ldlt_factorization is only implemented for consistent block tree."); // LCOV_EXCL_LINE
    }
    if (symmetry != 'S' and symmetry != 'H') {
        htool::Logger::get_instance().log(LogLevel::ERROR, "ldlt_factorization: symmetry=" + std::string(1, symmetry) + ", expected S or H."); // LCOV_EXCL_LINE
    }
    if (UPLO != 'L' and UPLO != 'U') {
        htool::Logger::get_instance().log(LogLevel::ERROR, "ldlt_factorization: UPLO=" + std::string(1, UPLO) + ", expected L or U."); // LCOV_EXCL_LINE
    }
    if (hmatrix.get_symmetry() != 'N' and hmatrix.get_symmetry() != symmetry and is_complex<CoefficientPrecision>()) {
        htool::Logger::get_instance().log(LogLevel::ERROR, "ldlt_factorization: symmetry=" + std::string(1, symmetry) + " does not match the HMatrix symmetry=" + std::string(1, hmatrix.get_symmetry()) + "."); // LCOV_EXCL_LINE
    }

#if __cplusplus >= 201703L
    if constexpr (is_execution_policy_v<std::decay_t<ExecutionPolicy>>) {
        if constexpr (std::is_same_v<std::decay_t<ExecutionPolicy>, exec_compat::parallel_policy>) {
            HMatrixTaskDependencies<CoefficientPrecision, CoordinatePrecision> hmatrix_task_dependencies;
            hmatrix_task_dependencies.set_L0(hmatrix);

#    if defined(_OPENMP)
#        pragma omp parallel
#        pragma omp single
#    endif
            {
                task_based_ldlt_factorization(symmetry, UPLO, hmatrix, hmatrix_task_dependencies.L0);
            }
        } else if constexpr (std::is_same_v<std::decay_t<ExecutionPolicy>, exec_compat::sequenced_policy>) {
            sequential_ldlt_factorization(symmetry, UPLO, hmatrix);
        } else if constexpr (std::is_same_v<std::decay_t<ExecutionPolicy>, omp_task_policy<CoefficientPrecision, CoordinatePrecision>>) {
            update_L0(execution_policy.hmatrix_task_dependencies, hmatrix);
            run_task_based(
                execution_policy.hmatrix_task_dependencies, hmatrix, [&]() { sequential_ldlt_factorization(symmetry, UPLO, hmatrix); }, [&](auto &L0) { task_based_ldlt_factorization(symmetry, UPLO, hmatrix, L0); });
        } else {
            static_assert(std::is_same_v<std::decay_t<ExecutionPolicy>, exec_compat::sequenced_policy> || std::is_same_v<std::decay_t<ExecutionPolicy>, exec_compat::parallel_policy> || std::is_same_v<std::decay_t<ExecutionPolicy>, omp_task_policy<CoefficientPrecision, CoordinatePrecision>>, "Invalid execution policy for factorization.");
        }
    } else {
        static_assert(is_execution_policy_v<std::decay_t<ExecutionPolicy>>, "Invalid execution policy for factorization.");
    }
#else
    (void)execution_policy;
    sequential_ldlt_factorization(symmetry, UPLO, hmatrix);
#endif
}

template <typename CoefficientPrecision, typename CoordinatePrecision = underlying_type<CoefficientPrecision>>
void ldlt_factorization(char symmetry, char UPLO, HMatrix<CoefficientPrecision, CoordinatePrecision> &hmatrix) {
#if __cplusplus >= 201703L
    ldlt_factorization(exec_compat::seq, symmetry, UPLO, hmatrix);
#else
    ldlt_factorization(nullptr, symmetry, UPLO, hmatrix);
#endif
}

} // namespace htool

#endif
