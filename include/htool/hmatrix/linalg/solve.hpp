#ifndef HTOOL_HMATRIX_LINALG_SOLVE_HPP
#define HTOOL_HMATRIX_LINALG_SOLVE_HPP

#include "../../clustering/cluster_node.hpp"              // for Cluster, user_to_cluster
#include "../../matrix/linalg/scale.hpp"                  // for scale
#include "../../matrix/linalg/transpose.hpp"              // for transpose
#include "../../matrix/matrix.hpp"                        // for Matrix
#include "../../misc/logger.hpp"                          // for Logger
#include "../../misc/misc.hpp"                            // for is_complex
#include "../execution_policies.hpp"                      // for omp_task_policy
#include "../hmatrix.hpp"                                 // for HMatrix
#include "apply_ldlt_diagonal.hpp"                        // for internal_apply_ldlt_diagonal
#include "task_based_apply_ldlt_diagonal.hpp"             // for task_based_internal_apply_ldlt_diagonal
#include "task_based_triangular_hmatrix_matrix_solve.hpp" // for task_based_internal_triangular_hmatrix_matrix_solve
#include "triangular_hmatrix_matrix_solve.hpp"            // for internal_triangular_hmatrix_matrix_solve
#include "triangular_ldlt_hmatrix_matrix_solve.hpp"       // for internal_triangular_ldlt_hmatrix_matrix_solve
#include <vector>                                         // for vector

// Solves with an HMatrix factorized by lu_factorization, cholesky_factorization or ldlt_factorization (see factorization.hpp).
// With an omp_task_policy, the solves reuse the L0 of the factorization and, unlike it, return once their tasks are done,
// also in a parallel region; there, their tasks run after the pending tasks of the factorization with the same policy.

namespace htool {

template <typename Mat, typename CoordinatePrecision = underlying_type<typename Mat::value_type>>
void internal_lu_solve(char trans, const HMatrix<typename Mat::value_type, CoordinatePrecision> &A, Mat &X) {
    using CoefficientPrecision = typename Mat::value_type;
    if (trans == 'N') {
        internal_triangular_hmatrix_matrix_solve('L', 'L', 'N', 'U', CoefficientPrecision(1), A, X);
        internal_triangular_hmatrix_matrix_solve('L', 'U', 'N', 'N', CoefficientPrecision(1), A, X);
    } else {
        internal_triangular_hmatrix_matrix_solve('L', 'U', trans, 'N', CoefficientPrecision(1), A, X);
        internal_triangular_hmatrix_matrix_solve('L', 'L', trans, 'U', CoefficientPrecision(1), A, X);
    }
}

template <typename CoefficientPrecision, typename CoordinatePrecision = underlying_type<CoefficientPrecision>>
void internal_cholesky_solve(char UPLO, const HMatrix<CoefficientPrecision, CoordinatePrecision> &A, Matrix<CoefficientPrecision> &X) {
    if (UPLO == 'L') {
        internal_triangular_hmatrix_matrix_solve('L', 'L', 'N', 'N', CoefficientPrecision(1), A, X);
        internal_triangular_hmatrix_matrix_solve('L', 'L', is_complex<CoefficientPrecision>() ? 'C' : 'T', 'N', CoefficientPrecision(1), A, X);
    } else {
        internal_triangular_hmatrix_matrix_solve('L', 'U', is_complex<CoefficientPrecision>() ? 'C' : 'T', 'N', CoefficientPrecision(1), A, X);
        internal_triangular_hmatrix_matrix_solve('L', 'U', 'N', 'N', CoefficientPrecision(1), A, X);
    }
}

// A = L D L^T (L^H for symmetry='H'), so X := L^-T D^-1 L^-1 X, as sytrs/hetrs.
template <typename CoefficientPrecision, typename CoordinatePrecision>
void internal_ldlt_solve(char symmetry, char UPLO, const HMatrix<CoefficientPrecision, CoordinatePrecision> &A, Matrix<CoefficientPrecision> &X) {
    internal_triangular_ldlt_hmatrix_matrix_solve('L', UPLO, 'N', CoefficientPrecision(1), A, X);
    internal_apply_ldlt_diagonal(symmetry, 'L', UPLO, A, X, A.get_target_cluster().get_offset());
    internal_triangular_ldlt_hmatrix_matrix_solve('L', UPLO, symmetry == 'H' ? 'C' : 'T', CoefficientPrecision(1), A, X);
}

// X := op(A)^-1 X, A holding its LU factors, on the row-major Xt = X^T (see task_based_triangular_hmatrix_matrix_solve.hpp).
// Only creates the tasks of the solve, so that in a parallel region it returns before they are done: Xt must stay alive and
// is only complete after a taskwait.
template <typename CoefficientPrecision, typename CoordinatePrecision>
void task_based_internal_lu_solve_row_major(char trans, const HMatrix<CoefficientPrecision, CoordinatePrecision> &A, Matrix<CoefficientPrecision> &Xt, const std::vector<const HMatrix<CoefficientPrecision, CoordinatePrecision> *> &L0) {
    std::vector<const Cluster<CoordinatePrecision> *> X_L0 = diagonal_clusters_of_L0(A, L0);
    // Cannot happen for a cut of a consistent block tree, as given by find_l0
    if (X_L0.empty()) {
        htool::Logger::get_instance().log(LogLevel::ERROR, "task_based_internal_lu_solve_row_major: the diagonal nodes of L0 do not tile the diagonal of A."); // LCOV_EXCL_LINE
        return;                                                                                                                                                // LCOV_EXCL_LINE
    }
    int mu       = Xt.nb_rows();
    int X_offset = A.get_target_cluster().get_offset();
    if (trans == 'N') {
        task_based_internal_triangular_hmatrix_matrix_solve_row_major('L', 'N', 'U', A, Xt.data(), mu, X_offset, L0, X_L0);
        task_based_internal_triangular_hmatrix_matrix_solve_row_major('U', 'N', 'N', A, Xt.data(), mu, X_offset, L0, X_L0);
    } else {
        task_based_internal_triangular_hmatrix_matrix_solve_row_major('U', trans, 'N', A, Xt.data(), mu, X_offset, L0, X_L0);
        task_based_internal_triangular_hmatrix_matrix_solve_row_major('L', trans, 'U', A, Xt.data(), mu, X_offset, L0, X_L0);
    }
}

// Same as task_based_internal_lu_solve_row_major on X, returning once the tasks are done.
template <typename CoefficientPrecision, typename CoordinatePrecision>
void task_based_internal_lu_solve(char trans, const HMatrix<CoefficientPrecision, CoordinatePrecision> &A, Matrix<CoefficientPrecision> &X, const std::vector<const HMatrix<CoefficientPrecision, CoordinatePrecision> *> &L0) {
    Matrix<CoefficientPrecision> Xt(X.nb_cols(), X.nb_rows());
    transpose(X, Xt);
    task_based_internal_lu_solve_row_major(trans, A, Xt, L0);
#if defined(_OPENMP)
#    pragma omp taskwait
#endif
    transpose(Xt, X);
}

// X := A^-1 X, A holding its Cholesky factor in its UPLO triangle, on the row-major Xt = X^T (see task_based_triangular_hmatrix_matrix_solve.hpp).
// Only creates the tasks of the solve, so that in a parallel region it returns before they are done: Xt must stay alive and
// is only complete after a taskwait.
template <typename CoefficientPrecision, typename CoordinatePrecision>
void task_based_internal_cholesky_solve_row_major(char UPLO, const HMatrix<CoefficientPrecision, CoordinatePrecision> &A, Matrix<CoefficientPrecision> &Xt, const std::vector<const HMatrix<CoefficientPrecision, CoordinatePrecision> *> &L0) {
    std::vector<const Cluster<CoordinatePrecision> *> X_L0 = diagonal_clusters_of_L0(A, L0);
    // Cannot happen for a cut of a consistent block tree, as given by find_l0
    if (X_L0.empty()) {
        htool::Logger::get_instance().log(LogLevel::ERROR, "task_based_internal_cholesky_solve_row_major: the diagonal nodes of L0 do not tile the diagonal of A."); // LCOV_EXCL_LINE
        return;                                                                                                                                                      // LCOV_EXCL_LINE
    }
    int mu       = Xt.nb_rows();
    int X_offset = A.get_target_cluster().get_offset();
    char trans   = is_complex<CoefficientPrecision>() ? 'C' : 'T';
    if (UPLO == 'L') {
        task_based_internal_triangular_hmatrix_matrix_solve_row_major('L', 'N', 'N', A, Xt.data(), mu, X_offset, L0, X_L0);
        task_based_internal_triangular_hmatrix_matrix_solve_row_major('L', trans, 'N', A, Xt.data(), mu, X_offset, L0, X_L0);
    } else {
        task_based_internal_triangular_hmatrix_matrix_solve_row_major('U', trans, 'N', A, Xt.data(), mu, X_offset, L0, X_L0);
        task_based_internal_triangular_hmatrix_matrix_solve_row_major('U', 'N', 'N', A, Xt.data(), mu, X_offset, L0, X_L0);
    }
}

// Same as task_based_internal_cholesky_solve_row_major on X, returning once the tasks are done.
template <typename CoefficientPrecision, typename CoordinatePrecision>
void task_based_internal_cholesky_solve(char UPLO, const HMatrix<CoefficientPrecision, CoordinatePrecision> &A, Matrix<CoefficientPrecision> &X, const std::vector<const HMatrix<CoefficientPrecision, CoordinatePrecision> *> &L0) {
    Matrix<CoefficientPrecision> Xt(X.nb_cols(), X.nb_rows());
    transpose(X, Xt);
    task_based_internal_cholesky_solve_row_major(UPLO, A, Xt, L0);
#if defined(_OPENMP)
#    pragma omp taskwait
#endif
    transpose(Xt, X);
}

// X := L^-T D^-1 L^-1 X (L^-H for symmetry='H'), A holding its LDLt factors, on the row-major Xt = X^T (see task_based_triangular_hmatrix_matrix_solve.hpp).
// Only creates the tasks of the solve, so that in a parallel region it returns before they are done: Xt must stay alive and
// is only complete after a taskwait.
template <typename CoefficientPrecision, typename CoordinatePrecision>
void task_based_internal_ldlt_solve_row_major(char symmetry, char UPLO, const HMatrix<CoefficientPrecision, CoordinatePrecision> &A, Matrix<CoefficientPrecision> &Xt, const std::vector<const HMatrix<CoefficientPrecision, CoordinatePrecision> *> &L0) {
    std::vector<const Cluster<CoordinatePrecision> *> X_L0 = diagonal_clusters_of_L0(A, L0);
    // Cannot happen for a cut of a consistent block tree, as given by find_l0
    if (X_L0.empty()) {
        htool::Logger::get_instance().log(LogLevel::ERROR, "task_based_internal_ldlt_solve_row_major: the diagonal nodes of L0 do not tile the diagonal of A."); // LCOV_EXCL_LINE
        return;                                                                                                                                                  // LCOV_EXCL_LINE
    }
    int mu       = Xt.nb_rows();
    int X_offset = A.get_target_cluster().get_offset();
    task_based_internal_triangular_ldlt_hmatrix_matrix_solve_row_major(UPLO, 'N', A, Xt.data(), mu, X_offset, L0, X_L0);
    task_based_internal_apply_ldlt_diagonal_row_major(symmetry, UPLO, Xt.data(), mu, X_offset, L0);
    task_based_internal_triangular_ldlt_hmatrix_matrix_solve_row_major(UPLO, symmetry == 'H' ? 'C' : 'T', A, Xt.data(), mu, X_offset, L0, X_L0);
}

// Same as task_based_internal_ldlt_solve_row_major on X, returning once the tasks are done.
template <typename CoefficientPrecision, typename CoordinatePrecision>
void task_based_internal_ldlt_solve(char symmetry, char UPLO, const HMatrix<CoefficientPrecision, CoordinatePrecision> &A, Matrix<CoefficientPrecision> &X, const std::vector<const HMatrix<CoefficientPrecision, CoordinatePrecision> *> &L0) {
    Matrix<CoefficientPrecision> Xt(X.nb_cols(), X.nb_rows());
    transpose(X, Xt);
    task_based_internal_ldlt_solve_row_major(symmetry, UPLO, A, Xt, L0);
#if defined(_OPENMP)
#    pragma omp taskwait
#endif
    transpose(Xt, X);
}

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

// The single triangular solves and the application of D^-1 of the solves above, used by LocalHMatrixPlusOverlapSolver.
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
