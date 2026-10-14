#ifndef HTOOL_HMATRIX_LINALG_SOLVE_HPP
#define HTOOL_HMATRIX_LINALG_SOLVE_HPP

#include "../../clustering/cluster_node.hpp"
#include "../../matrix/matrix.hpp"
#include "../../misc/misc.hpp"
#include "../execution_policies.hpp"
#include "../hmatrix.hpp"
#include "omp_task/solve.hpp"
#include "seq/solve.hpp"
#include <vector>

// Solves with an HMatrix factorized by lu_factorization, cholesky_factorization or ldlt_factorization (see factorization.hpp).
// With an omp_task_policy, the solves reuse the L0 of the factorization and, unlike it, return once their tasks are done,
// also in a parallel region; there, their tasks run after the pending tasks of the factorization with the same policy.

namespace htool {

template <typename ExecutionPolicy, typename CoefficientPrecision, typename CoordinatePrecision>
void internal_lu_solve(ExecutionPolicy &&execution_policy, char trans, const HMatrix<CoefficientPrecision, CoordinatePrecision> &A, Matrix<CoefficientPrecision> &X) {
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
                task_based_internal_lu_solve(trans, A, X, hmatrix_task_dependencies.L0);
            }
        } else if constexpr (std::is_same_v<std::decay_t<ExecutionPolicy>, exec_compat::sequenced_policy>) {
            internal_lu_solve(trans, A, X);
        } else if constexpr (std::is_same_v<std::decay_t<ExecutionPolicy>, omp_task_policy<CoefficientPrecision, CoordinatePrecision>>) {
            update_L0(execution_policy.hmatrix_task_dependencies, A);
            run_task_based(
                execution_policy.hmatrix_task_dependencies, A, [&]() { internal_lu_solve(trans, A, X); }, [&](auto &L0) { task_based_internal_lu_solve(trans, A, X, L0); });
        } else {
            static_assert(std::is_same_v<std::decay_t<ExecutionPolicy>, exec_compat::sequenced_policy> || std::is_same_v<std::decay_t<ExecutionPolicy>, exec_compat::parallel_policy> || std::is_same_v<std::decay_t<ExecutionPolicy>, omp_task_policy<CoefficientPrecision, CoordinatePrecision>>, "Invalid execution policy for solve.");
        }
    } else {
        static_assert(is_execution_policy_v<std::decay_t<ExecutionPolicy>>, "Invalid execution policy for solve.");
    }
#else
    internal_lu_solve(trans, A, X);
#endif
}

template <typename ExecutionPolicy, typename CoefficientPrecision, typename CoordinatePrecision>
void internal_cholesky_solve(ExecutionPolicy &&execution_policy, char UPLO, const HMatrix<CoefficientPrecision, CoordinatePrecision> &A, Matrix<CoefficientPrecision> &X) {
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
                task_based_internal_cholesky_solve(UPLO, A, X, hmatrix_task_dependencies.L0);
            }
        } else if constexpr (std::is_same_v<std::decay_t<ExecutionPolicy>, exec_compat::sequenced_policy>) {
            internal_cholesky_solve(UPLO, A, X);
        } else if constexpr (std::is_same_v<std::decay_t<ExecutionPolicy>, omp_task_policy<CoefficientPrecision, CoordinatePrecision>>) {
            update_L0(execution_policy.hmatrix_task_dependencies, A);
            run_task_based(
                execution_policy.hmatrix_task_dependencies, A, [&]() { internal_cholesky_solve(UPLO, A, X); }, [&](auto &L0) { task_based_internal_cholesky_solve(UPLO, A, X, L0); });
        } else {
            static_assert(std::is_same_v<std::decay_t<ExecutionPolicy>, exec_compat::sequenced_policy> || std::is_same_v<std::decay_t<ExecutionPolicy>, exec_compat::parallel_policy> || std::is_same_v<std::decay_t<ExecutionPolicy>, omp_task_policy<CoefficientPrecision, CoordinatePrecision>>, "Invalid execution policy for solve.");
        }
    } else {
        static_assert(is_execution_policy_v<std::decay_t<ExecutionPolicy>>, "Invalid execution policy for solve.");
    }
#else
    internal_cholesky_solve(UPLO, A, X);
#endif
}

template <typename ExecutionPolicy, typename CoefficientPrecision, typename CoordinatePrecision>
void internal_ldlt_solve(ExecutionPolicy &&execution_policy, char symmetry, char UPLO, const HMatrix<CoefficientPrecision, CoordinatePrecision> &A, Matrix<CoefficientPrecision> &X) {
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
                task_based_internal_ldlt_solve(symmetry, UPLO, A, X, hmatrix_task_dependencies.L0);
            }
        } else if constexpr (std::is_same_v<std::decay_t<ExecutionPolicy>, exec_compat::sequenced_policy>) {
            internal_ldlt_solve(symmetry, UPLO, A, X);
        } else if constexpr (std::is_same_v<std::decay_t<ExecutionPolicy>, omp_task_policy<CoefficientPrecision, CoordinatePrecision>>) {
            update_L0(execution_policy.hmatrix_task_dependencies, A);
            run_task_based(
                execution_policy.hmatrix_task_dependencies, A, [&]() { internal_ldlt_solve(symmetry, UPLO, A, X); }, [&](auto &L0) { task_based_internal_ldlt_solve(symmetry, UPLO, A, X, L0); });
        } else {
            static_assert(std::is_same_v<std::decay_t<ExecutionPolicy>, exec_compat::sequenced_policy> || std::is_same_v<std::decay_t<ExecutionPolicy>, exec_compat::parallel_policy> || std::is_same_v<std::decay_t<ExecutionPolicy>, omp_task_policy<CoefficientPrecision, CoordinatePrecision>>, "Invalid execution policy for solve.");
        }
    } else {
        static_assert(is_execution_policy_v<std::decay_t<ExecutionPolicy>>, "Invalid execution policy for solve.");
    }
#else
    internal_ldlt_solve(symmetry, UPLO, A, X);
#endif
}

template <typename ExecutionPolicy, typename Mat, typename CoordinatePrecision = underlying_type<typename Mat::value_type>>
void lu_solve(ExecutionPolicy &&execution_policy, char trans, const HMatrix<typename Mat::value_type, CoordinatePrecision> &A, Mat &X) {
    using CoefficientPrecision = typename Mat::value_type;
    Matrix<CoefficientPrecision> permuted_X(X.nb_rows(), X.nb_cols());
    auto &source_cluster = A.get_source_cluster();
    for (int i = 0; i < X.nb_cols(); i++) {
        user_to_cluster(source_cluster, X.data() + source_cluster.get_size() * i, permuted_X.data() + source_cluster.get_size() * i);
    }

    internal_lu_solve(std::forward<ExecutionPolicy>(execution_policy), trans, A, permuted_X);

    auto &target_cluster = A.get_target_cluster();
    for (int i = 0; i < X.nb_cols(); i++) {
        cluster_to_user(target_cluster, permuted_X.data() + target_cluster.get_size() * i, X.data() + target_cluster.get_size() * i);
    }
}

template <typename ExecutionPolicy, typename Mat, typename CoordinatePrecision = underlying_type<typename Mat::value_type>>
void cholesky_solve(ExecutionPolicy &&execution_policy, char UPLO, const HMatrix<typename Mat::value_type, CoordinatePrecision> &A, Mat &X) {
    using CoefficientPrecision = typename Mat::value_type;
    Matrix<CoefficientPrecision> permuted_X(X.nb_rows(), X.nb_cols());
    auto &source_cluster = A.get_source_cluster();
    for (int i = 0; i < X.nb_cols(); i++) {
        user_to_cluster(source_cluster, X.data() + source_cluster.get_size() * i, permuted_X.data() + source_cluster.get_size() * i);
    }

    internal_cholesky_solve(std::forward<ExecutionPolicy>(execution_policy), UPLO, A, permuted_X);

    auto &target_cluster = A.get_target_cluster();
    for (int i = 0; i < X.nb_cols(); i++) {
        cluster_to_user(target_cluster, permuted_X.data() + target_cluster.get_size() * i, X.data() + target_cluster.get_size() * i);
    }
}

template <typename ExecutionPolicy, typename Mat, typename CoordinatePrecision = underlying_type<typename Mat::value_type>>
void ldlt_solve(ExecutionPolicy &&execution_policy, char symmetry, char UPLO, const HMatrix<typename Mat::value_type, CoordinatePrecision> &A, Mat &X) {
    using CoefficientPrecision = typename Mat::value_type;
    Matrix<CoefficientPrecision> permuted_X(X.nb_rows(), X.nb_cols());
    auto &source_cluster = A.get_source_cluster();
    for (int i = 0; i < X.nb_cols(); i++) {
        user_to_cluster(source_cluster, X.data() + source_cluster.get_size() * i, permuted_X.data() + source_cluster.get_size() * i);
    }

    internal_ldlt_solve(std::forward<ExecutionPolicy>(execution_policy), symmetry, UPLO, A, permuted_X);

    auto &target_cluster = A.get_target_cluster();
    for (int i = 0; i < X.nb_cols(); i++) {
        cluster_to_user(target_cluster, permuted_X.data() + target_cluster.get_size() * i, X.data() + target_cluster.get_size() * i);
    }
}

template <typename Mat, typename CoordinatePrecision = underlying_type<typename Mat::value_type>>
void lu_solve(char trans, const HMatrix<typename Mat::value_type, CoordinatePrecision> &A, Mat &X) {
    lu_solve(exec_compat::seq, trans, A, X);
}

template <typename Mat, typename CoordinatePrecision = underlying_type<typename Mat::value_type>>
void cholesky_solve(char UPLO, const HMatrix<typename Mat::value_type, CoordinatePrecision> &A, Mat &X) {
    cholesky_solve(exec_compat::seq, UPLO, A, X);
}

template <typename Mat, typename CoordinatePrecision = underlying_type<typename Mat::value_type>>
void ldlt_solve(char symmetry, char UPLO, const HMatrix<typename Mat::value_type, CoordinatePrecision> &A, Mat &X) {
    ldlt_solve(exec_compat::seq, symmetry, UPLO, A, X);
}

} // namespace htool

#endif
