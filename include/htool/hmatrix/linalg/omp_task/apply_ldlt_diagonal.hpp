#ifndef HTOOL_HMATRIX_LINALG_OMP_TASK_APPLY_LDLT_DIAGONAL_HPP
#define HTOOL_HMATRIX_LINALG_OMP_TASK_APPLY_LDLT_DIAGONAL_HPP

#include "../../../clustering/cluster_node.hpp"
#include "../../../matrix/linalg/transpose.hpp"
#include "../../../matrix/matrix.hpp"
#include "../../../misc/logger.hpp"
#include "../../hmatrix.hpp"
#include "../../task_dependencies.hpp"
#include "../seq/apply_ldlt_diagonal.hpp"
#include "triangular_hmatrix_matrix_solve.hpp" // for diagonal_clusters_of_L0
#include <vector>

// to remove warning from depend(iterator(it = 0 : read_deps_size), in : *read_deps[it])
#if defined(__clang__)
#elif defined(__GNUC__) || defined(__GNUG__)
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wuseless-cast"
#endif

namespace htool {

/**
 * @brief Task-based internal_apply_ldlt_diagonal: B := D^-1 B (side='L') or B D^-1 (side='R'), with one task per node of L0_B in B.
 *
 * @param[in] A The HMatrix holding D at its dense diagonal leaves.
 * @param[in,out] B The HMatrix \f$B\f$.
 * @param[in] L0_A The L0 of \f$A\f$.
 * @param[in] L0_B The L0 of \f$B\f$.
 */
template <typename CoefficientPrecision, typename CoordinatePrecision>
void task_based_internal_apply_ldlt_diagonal(char symmetry, char side, char UPLO, const HMatrix<CoefficientPrecision, CoordinatePrecision> &A, HMatrix<CoefficientPrecision, CoordinatePrecision> &B, const std::vector<const HMatrix<CoefficientPrecision, CoordinatePrecision> *> &L0_A, const std::vector<const HMatrix<CoefficientPrecision, CoordinatePrecision> *> &L0_B) {
    for (const HMatrix<CoefficientPrecision, CoordinatePrecision> *L0_node : enumerate_dependences(B, L0_B)) {
        // L0_node not in B's block tree (B below L0_B): the task only modifies B
        HMatrix<CoefficientPrecision, CoordinatePrecision> *B_node = get_mutable_node(B, *L0_node);
        if (B_node == nullptr) {
            B_node = &B;
        }

        const Cluster<CoordinatePrecision> &cluster                      = side == 'L' ? B_node->get_target_cluster() : B_node->get_source_cluster();
        const HMatrix<CoefficientPrecision, CoordinatePrecision> *A_node = A.get_sub_hmatrix(cluster, cluster);
        // Cannot happen for a consistent block tree, whose diagonal blocks are refined down to the leaf clusters
        if (A_node == nullptr) {
            htool::Logger::get_instance().log(LogLevel::ERROR, "task_based_internal_apply_ldlt_diagonal: no diagonal block of A matches a block of B in L0."); // LCOV_EXCL_LINE
            continue;                                                                                                                                          // LCOV_EXCL_LINE
        }

#if defined(_OPENMP)
        std::vector<const HMatrix<CoefficientPrecision, CoordinatePrecision> *> read_deps = enumerate_dependences(*A_node, L0_A);
        int read_deps_size                                                                = read_deps.size();
#    pragma omp task default(none)                         \
        firstprivate(symmetry, side, UPLO, A_node, B_node) \
        shared(read_deps)                                  \
        depend(inout : *L0_node)                           \
        depend(iterator(it = 0 : read_deps_size), in : *read_deps[it])
#else
        (void)L0_A;
#endif
        {
            internal_apply_ldlt_diagonal(symmetry, side, UPLO, *A_node, *B_node);
        }
    }
}

/**
 * @brief Task-based internal_apply_ldlt_diagonal with side='L' on the row-major right-hand side Xt (see omp_task/triangular_hmatrix_matrix_solve.hpp): X := D^-1 X, with one task per diagonal node of L0_A.
 */
template <typename CoefficientPrecision, typename CoordinatePrecision>
void task_based_internal_apply_ldlt_diagonal_row_major(char symmetry, char UPLO, CoefficientPrecision *Xt, int mu, int X_offset, const std::vector<const HMatrix<CoefficientPrecision, CoordinatePrecision> *> &L0_A, const std::vector<const Cluster<CoordinatePrecision> *> &X_L0) {
    for (const HMatrix<CoefficientPrecision, CoordinatePrecision> *A_node : L0_A) {
        const Cluster<CoordinatePrecision> *X_node = &A_node->get_target_cluster();
        if (!(*X_node == A_node->get_source_cluster())) {
            continue;
        }
        int size                      = X_node->get_size();
        int offset                    = X_node->get_offset();
        CoefficientPrecision *Xt_node = Xt + mu * (offset - X_offset);
#if defined(_OPENMP)
        std::vector<CoefficientPrecision *> deps = blocks_of_rows(Xt, mu, X_offset, *X_node, X_L0);
        int deps_size                            = deps.size();
#    pragma omp task default(none)                                      \
        firstprivate(symmetry, UPLO, A_node, Xt_node, mu, size, offset) \
        shared(deps)                                                    \
        depend(in : *A_node)                                            \
        depend(iterator(it = 0 : deps_size), inout : *deps[it])
#else
        (void)X_L0;
#endif
        {
            // D^-1 X on column-major rows: (D^-1 X)^T is not Xt D^-1 for a Hermitian D
            Matrix<CoefficientPrecision> Bt, B(size, mu);
            Bt.assign(mu, size, Xt_node, false);
            transpose(Bt, B);
            internal_apply_ldlt_diagonal(symmetry, 'L', UPLO, *A_node, B, offset);
            transpose(B, Bt);
        }
    }
}
// Same as internal_apply_ldlt_diagonal on a Matrix B, D being held by the diagonal nodes of L0, with one task per diagonal node of L0, returning once they are done.
// Each task works on the rows (side='L') or columns (side='R') of B of its node, and depends on the blocks of them given by X_L0,
// a cut of the cluster tree of the rows of A (see omp_task/triangular_hmatrix_matrix_solve.hpp), through the address of their first coefficient.
template <typename CoefficientPrecision, typename CoordinatePrecision>
void task_based_internal_apply_ldlt_diagonal(char symmetry, char side, char UPLO, const HMatrix<CoefficientPrecision, CoordinatePrecision> &A, Matrix<CoefficientPrecision> &B, int offset, const std::vector<const HMatrix<CoefficientPrecision, CoordinatePrecision> *> &L0, const std::vector<const Cluster<CoordinatePrecision> *> &X_L0) {
    // Cannot happen for a cut of a consistent block tree, as given by find_l0
    if (diagonal_clusters_of_L0(A, L0).empty()) {
        htool::Logger::get_instance().log(LogLevel::ERROR, "task_based_internal_apply_ldlt_diagonal: the diagonal nodes of L0 do not tile the diagonal of A."); // LCOV_EXCL_LINE
        return;                                                                                                                                                 // LCOV_EXCL_LINE
    }
    for (const HMatrix<CoefficientPrecision, CoordinatePrecision> *A_node : L0) {
        if (!(A_node->get_target_cluster() == A_node->get_source_cluster())) {
            continue;
        }
#if defined(_OPENMP)
        // The rows (side='L', with stride 1) or columns (side='R', with stride B.nb_rows()) of B start at the coefficient offset
        std::vector<CoefficientPrecision *> deps = blocks_of_rows(B.data(), side == 'L' ? 1 : B.nb_rows(), offset, A_node->get_target_cluster(), X_L0);
        int deps_size                            = deps.size();
#    pragma omp task default(none)                         \
        firstprivate(symmetry, side, UPLO, A_node, offset) \
        shared(B, deps)                                    \
        depend(in : *A_node)                               \
        depend(iterator(it = 0 : deps_size), inout : *deps[it])
#else
        (void)X_L0;
#endif
        {
            internal_apply_ldlt_diagonal(symmetry, side, UPLO, *A_node, B, offset);
        }
    }
#if defined(_OPENMP)
#    pragma omp taskwait
#endif
}

} // namespace htool

#if defined(__clang__)
#elif defined(__GNUC__) || defined(__GNUG__)
#    pragma GCC diagnostic pop
#endif
#endif
