#ifndef HTOOL_TASK_BASED_HMATRIX_LINALG_TRIANGULAR_HMATRIX_MATRIX_SOLVE_HPP
#define HTOOL_TASK_BASED_HMATRIX_LINALG_TRIANGULAR_HMATRIX_MATRIX_SOLVE_HPP

#include "../../clustering/cluster_node.hpp"          // for Cluster
#include "../../matrix/linalg/transpose.hpp"          // for transpose
#include "../../matrix/matrix.hpp"                    // for Matrix
#include "../hmatrix.hpp"                             // for HMatrix
#include "../task_dependencies.hpp"                   // for enumerate_dependences
#include "add_hmatrix_matrix_product_row_major.hpp"   // for internal_add_hmatrix_matrix_product_row_major
#include "apply_ldlt_diagonal.hpp"                    // for internal_apply_ldlt_diagonal
#include "task_based_add_hmatrix_hmatrix_product.hpp" // for fill_clusters
#include "triangular_hmatrix_matrix_solve.hpp"        // for internal_triangular_hmatrix_matrix_solve_row_major
#include "triangular_ldlt_hmatrix_matrix_solve.hpp"   // for internal_triangular_ldlt_hmatrix_matrix_solve_row_major
#include <algorithm>                                  // for find, sort
#include <vector>                                     // for vector

// to remove warning from depend(iterator(it = 0 : read_deps_size), in : *read_deps[it])
#if defined(__clang__)
#elif defined(__GNUC__) || defined(__GNUG__)
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wuseless-cast"
#endif

namespace htool {

// The task-based solves below act on a dense right-hand side stored row-major: Xt is mu x n, the column of Xt
// for the row i of X (numbered from X_offset) being Xt + mu * (i - X_offset). The tasks depend on the rows of X
// through X_L0, the target clusters of the diagonal nodes of L0_A, which tile the rows of A.

// Returns X_L0 for A and L0_A, sorted by offset, or an empty vector if the diagonal nodes of L0_A do not tile
// the rows of A (e.g. for a block tree that is not consistent).
template <typename CoefficientPrecision, typename CoordinatePrecision>
std::vector<const Cluster<CoordinatePrecision> *> diagonal_clusters_of_L0(const HMatrix<CoefficientPrecision, CoordinatePrecision> &A, const std::vector<const HMatrix<CoefficientPrecision, CoordinatePrecision> *> &L0_A) {
    std::vector<const Cluster<CoordinatePrecision> *> X_L0;
    for (auto &L0_node : L0_A) {
        if (L0_node->get_target_cluster() == L0_node->get_source_cluster()) {
            X_L0.push_back(&L0_node->get_target_cluster());
        }
    }
    std::sort(X_L0.begin(), X_L0.end(), [](const Cluster<CoordinatePrecision> *a, const Cluster<CoordinatePrecision> *b) { return a->get_offset() < b->get_offset(); });
    int offset = A.get_target_cluster().get_offset();
    for (auto &cluster : X_L0) {
        if (cluster->get_offset() != offset) {
            return {}; // LCOV_EXCL_LINE
        }
        offset += cluster->get_size();
    }
    if (offset != A.get_target_cluster().get_offset() + A.get_target_cluster().get_size()) {
        return {}; // LCOV_EXCL_LINE
    }
    return X_L0;
}

// X_output -= op(A_child) X_input in a triangular solve, with one task per node of L0_A in A_child.
template <typename CoefficientPrecision, typename CoordinatePrecision>
void task_based_internal_triangular_solve_update(char transa, const HMatrix<CoefficientPrecision, CoordinatePrecision> &A_child, CoefficientPrecision *Xt, int mu, int X_offset, const std::vector<const HMatrix<CoefficientPrecision, CoordinatePrecision> *> &L0_A, const std::vector<const Cluster<CoordinatePrecision> *> &X_L0) {
    for (const HMatrix<CoefficientPrecision, CoordinatePrecision> *L0_node : enumerate_dependences(A_child, L0_A)) {
        const Cluster<CoordinatePrecision> &output = transa == 'N' ? L0_node->get_target_cluster() : L0_node->get_source_cluster();
        const Cluster<CoordinatePrecision> &input  = transa == 'N' ? L0_node->get_source_cluster() : L0_node->get_target_cluster();
        const CoefficientPrecision *Xt_input       = Xt + mu * (input.get_offset() - X_offset);
        CoefficientPrecision *Xt_output            = Xt + mu * (output.get_offset() - X_offset);
#if defined(_OPENMP)
        std::vector<const Cluster<CoordinatePrecision> *> write_deps = enumerate_dependences(output, X_L0), read_deps = enumerate_dependences(input, X_L0);
        int write_deps_size = write_deps.size(), read_deps_size = read_deps.size();
#    pragma omp task default(none)                                          \
        firstprivate(transa, L0_node, Xt_input, Xt_output, mu)              \
        shared(write_deps, read_deps)                                       \
        depend(in : *L0_node)                                               \
        depend(iterator(it = 0 : write_deps_size), inout : *write_deps[it]) \
        depend(iterator(it = 0 : read_deps_size), in : *read_deps[it])
#else
        (void)X_L0;
#endif
        {
            internal_add_hmatrix_matrix_product_row_major(transa, 'N', CoefficientPrecision(-1), *L0_node, Xt_input, CoefficientPrecision(1), Xt_output, mu);
        }
    }
}

// Creates the tasks of a triangular solve with side='L', in the order of internal_triangular_hmatrix_matrix_solve_row_major,
// leaf_solve(UPLO, transa, A, Bt) being the sequential row-major solve applied at the diagonal nodes of L0_A.
template <typename LeafSolve, typename CoefficientPrecision, typename CoordinatePrecision>
void task_based_internal_triangular_hmatrix_matrix_solve_row_major_with(LeafSolve leaf_solve, char UPLO, char transa, const HMatrix<CoefficientPrecision, CoordinatePrecision> &A, CoefficientPrecision *Xt, int mu, int X_offset, const std::vector<const HMatrix<CoefficientPrecision, CoordinatePrecision> *> &L0_A, const std::vector<const Cluster<CoordinatePrecision> *> &X_L0) {
    if (std::find(L0_A.begin(), L0_A.end(), &A) != L0_A.end()) {
        const HMatrix<CoefficientPrecision, CoordinatePrecision> *A_node = &A;
        const Cluster<CoordinatePrecision> *X_node                       = &A.get_target_cluster();
        int size                                                         = X_node->get_size();
        CoefficientPrecision *Xt_node                                    = Xt + mu * (X_node->get_offset() - X_offset);
#if defined(_OPENMP)
#    pragma omp task default(none)                                        \
        firstprivate(leaf_solve, UPLO, transa, A_node, Xt_node, mu, size) \
        depend(in : *A_node)                                              \
        depend(inout : *X_node)
#endif
        {
            Matrix<CoefficientPrecision> Bt;
            Bt.assign(mu, size, Xt_node, false);
            leaf_solve(UPLO, transa, *A_node, Bt);
        }
        return;
    }

    bool block_tree_not_consistent = (A.get_target_cluster().get_rank() < 0 || A.get_source_cluster().get_rank() < 0);
    std::vector<const Cluster<CoordinatePrecision> *> output_clusters, input_clusters;
    const Cluster<CoordinatePrecision> &output_cluster = transa == 'N' ? A.get_target_cluster() : A.get_source_cluster();
    const Cluster<CoordinatePrecision> &input_cluster  = transa == 'N' ? A.get_source_cluster() : A.get_target_cluster();
    fill_clusters(output_cluster, block_tree_not_consistent, output_clusters);
    fill_clusters(input_cluster, block_tree_not_consistent, input_clusters);

    // Forward
    if ((UPLO == 'L' && transa == 'N') || (UPLO == 'U' && transa != 'N')) {
        for (auto &output_cluster_child : output_clusters) {
            for (auto &input_cluster_child : input_clusters) {
                const HMatrix<CoefficientPrecision, CoordinatePrecision> *A_child = transa == 'N' ? A.get_sub_hmatrix(*output_cluster_child, *input_cluster_child) : A.get_sub_hmatrix(*input_cluster_child, *output_cluster_child);
                if (*output_cluster_child == *input_cluster_child) {
                    task_based_internal_triangular_hmatrix_matrix_solve_row_major_with(leaf_solve, UPLO, transa, *A_child, Xt, mu, X_offset, L0_A, X_L0);
                } else if (output_cluster_child->get_offset() > input_cluster_child->get_offset()) {
                    task_based_internal_triangular_solve_update(transa, *A_child, Xt, mu, X_offset, L0_A, X_L0);
                }
            }
        }
    } // Backward
    else {
        for (auto output_it = output_clusters.rbegin(); output_it != output_clusters.rend(); ++output_it) {
            for (auto input_it = input_clusters.rbegin(); input_it != input_clusters.rend(); ++input_it) {
                const HMatrix<CoefficientPrecision, CoordinatePrecision> *A_child = transa == 'N' ? A.get_sub_hmatrix(**output_it, **input_it) : A.get_sub_hmatrix(**input_it, **output_it);
                if (**output_it == **input_it) {
                    task_based_internal_triangular_hmatrix_matrix_solve_row_major_with(leaf_solve, UPLO, transa, *A_child, Xt, mu, X_offset, L0_A, X_L0);
                } else if ((*output_it)->get_offset() < (*input_it)->get_offset()) {
                    task_based_internal_triangular_solve_update(transa, *A_child, Xt, mu, X_offset, L0_A, X_L0);
                }
            }
        }
    }
}

/**
 * @brief Task-based internal_triangular_hmatrix_matrix_solve with side='L' and alpha=1, on the row-major right-hand side Xt (see above).
 *
 * @param[in] A The triangular HMatrix \f$A\f$.
 * @param[in,out] Xt The right-hand side, row-major, overwritten by the solution.
 * @param[in] L0_A The L0 of \f$A\f$.
 * @param[in] X_L0 The clusters given by diagonal_clusters_of_L0(A, L0_A).
 */
template <typename CoefficientPrecision, typename CoordinatePrecision>
void task_based_internal_triangular_hmatrix_matrix_solve_row_major(char UPLO, char transa, char diag, const HMatrix<CoefficientPrecision, CoordinatePrecision> &A, CoefficientPrecision *Xt, int mu, int X_offset, const std::vector<const HMatrix<CoefficientPrecision, CoordinatePrecision> *> &L0_A, const std::vector<const Cluster<CoordinatePrecision> *> &X_L0) {
    task_based_internal_triangular_hmatrix_matrix_solve_row_major_with(
        [diag](char leaf_UPLO, char leaf_transa, const HMatrix<CoefficientPrecision, CoordinatePrecision> &leaf_A, Matrix<CoefficientPrecision> &leaf_Bt) { internal_triangular_hmatrix_matrix_solve_row_major('L', leaf_UPLO, leaf_transa, diag, CoefficientPrecision(1), leaf_A, leaf_Bt); }, UPLO, transa, A, Xt, mu, X_offset, L0_A, X_L0);
}

/**
 * @brief Same as task_based_internal_triangular_hmatrix_matrix_solve_row_major, solving with the unit \f$L\f$ or \f$U\f$ of an LDLt factorization.
 */
template <typename CoefficientPrecision, typename CoordinatePrecision>
void task_based_internal_triangular_ldlt_hmatrix_matrix_solve_row_major(char UPLO, char transa, const HMatrix<CoefficientPrecision, CoordinatePrecision> &A, CoefficientPrecision *Xt, int mu, int X_offset, const std::vector<const HMatrix<CoefficientPrecision, CoordinatePrecision> *> &L0_A, const std::vector<const Cluster<CoordinatePrecision> *> &X_L0) {
    task_based_internal_triangular_hmatrix_matrix_solve_row_major_with(
        [](char leaf_UPLO, char leaf_transa, const HMatrix<CoefficientPrecision, CoordinatePrecision> &leaf_A, Matrix<CoefficientPrecision> &leaf_Bt) { internal_triangular_ldlt_hmatrix_matrix_solve_row_major('L', leaf_UPLO, leaf_transa, CoefficientPrecision(1), leaf_A, leaf_Bt); }, UPLO, transa, A, Xt, mu, X_offset, L0_A, X_L0);
}

/**
 * @brief Task-based internal_apply_ldlt_diagonal with side='L' on the row-major right-hand side Xt (see above): X := D^-1 X, with one task per diagonal node of L0_A.
 */
template <typename CoefficientPrecision, typename CoordinatePrecision>
void task_based_internal_apply_ldlt_diagonal_row_major(char symmetry, char UPLO, CoefficientPrecision *Xt, int mu, int X_offset, const std::vector<const HMatrix<CoefficientPrecision, CoordinatePrecision> *> &L0_A) {
    for (const HMatrix<CoefficientPrecision, CoordinatePrecision> *A_node : L0_A) {
        const Cluster<CoordinatePrecision> *X_node = &A_node->get_target_cluster();
        if (!(*X_node == A_node->get_source_cluster())) {
            continue;
        }
        int size                      = X_node->get_size();
        int offset                    = X_node->get_offset();
        CoefficientPrecision *Xt_node = Xt + mu * (offset - X_offset);
#if defined(_OPENMP)
#    pragma omp task default(none)                                      \
        firstprivate(symmetry, UPLO, A_node, Xt_node, mu, size, offset) \
        depend(in : *A_node)                                            \
        depend(inout : *X_node)
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

} // namespace htool

#if defined(__clang__)
#elif defined(__GNUC__) || defined(__GNUG__)
#    pragma GCC diagnostic pop
#endif
#endif
