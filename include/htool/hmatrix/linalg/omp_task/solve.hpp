#ifndef HTOOL_HMATRIX_LINALG_OMP_TASK_SOLVE_HPP
#define HTOOL_HMATRIX_LINALG_OMP_TASK_SOLVE_HPP

#include "../../../clustering/cluster_node.hpp"
#include "../../../matrix/linalg/transpose.hpp"
#include "../../../matrix/matrix.hpp"
#include "../../../misc/logger.hpp"
#include "../../../misc/misc.hpp"
#include "../../hmatrix.hpp"
#include "apply_ldlt_diagonal.hpp"
#include "triangular_hmatrix_matrix_solve.hpp"
#include <vector>

namespace htool {

// X := op(A)^-1 X, A holding its LU factors, on the row-major Xt = X^T (see omp_task/triangular_hmatrix_matrix_solve.hpp).
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

// X := A^-1 X, A holding its Cholesky factor in its UPLO triangle, on the row-major Xt = X^T (see omp_task/triangular_hmatrix_matrix_solve.hpp).
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

// X := L^-T D^-1 L^-1 X (L^-H for symmetry='H'), A holding its LDLt factors, on the row-major Xt = X^T (see omp_task/triangular_hmatrix_matrix_solve.hpp).
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

} // namespace htool

#endif
