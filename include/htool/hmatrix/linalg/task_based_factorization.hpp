#ifndef HTOOL_TASK_BASED_HMATRIX_LINALG_FACTORIZATION_HPP
#define HTOOL_TASK_BASED_HMATRIX_LINALG_FACTORIZATION_HPP

#include "../../clustering/cluster_node.hpp" // for Cluster
#include "../../matrix/matrix.hpp"
#include "../../misc/logger.hpp" // for Logger
#include "../../misc/misc.hpp"   // for is_c...
#include "../hmatrix.hpp"        // for HMatrix
// #include "../linalg/triangular_hmatrix_hmatrix_solve.hpp"       // for tria...
// #include "../linalg/triangular_hmatrix_matrix_solve.hpp"        // for tria...
// #include "htool/hmatrix/linalg/add_hmatrix_hmatrix_product.hpp" // for add_...
#include "task_based_add_hmatrix_hmatrix_product.hpp"      // for add_hmatrix_hmatrix_product
#include "task_based_apply_ldlt_diagonal.hpp"              // for task_based_internal_apply_ldlt_diagonal
#include "task_based_triangular_hmatrix_hmatrix_solve.hpp" // for task_based_internal_triangular_hmatrix_hmatrix_solve
#include <algorithm>                                       // for reverse
#include <memory>                                          // for shared_ptr, unique_ptr
#include <string>                                          // for basi...
#include <vector>                                          // for vector

// to remove warning from depend(iterator(it = 0 : Ls_L0_nodes_size), inout : *Ls_L0_nodes[it])
#if defined(__clang__)
#elif defined(__GNUC__) || defined(__GNUG__)
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wuseless-cast"
#endif

namespace htool {

/**
 * @brief Task-based LU factorization of a hierarchical matrix.
 *
 * This function performs a task-based LU factorization of a hierarchical matrix. The matrix is
 * factorized as a product of a lower triangular matrix L and an upper triangular matrix U, i.e.
 * A = LU. The block structure of the matrix is used to split the computation into tasks that are
 * independent and can be executed concurrently.
 *
 * @param hmatrix The hierarchical matrix to be factorized.
 * @param L0 A vector of pointers to nodes of the output 'hmatrix'.
 *
 * @see lu_factorization
 */
template <typename CoefficientPrecision, typename CoordinatePrecision = underlying_type<CoefficientPrecision>>
void task_based_lu_factorization(HMatrix<CoefficientPrecision, CoordinatePrecision> &hmatrix, std::vector<HMatrix<CoefficientPrecision> *> &L0) {
    if (hmatrix.is_hierarchical()) {
        // check if hmatrix is in L0.
        bool is_hmatrix_in_L0 = false;
        for (auto &L0_node : L0) {
            if (L0_node->get_target_cluster() == hmatrix.get_target_cluster() && L0_node->get_source_cluster() == hmatrix.get_source_cluster()) {
                is_hmatrix_in_L0 = true;
                break;
            }
        }

        if (is_hmatrix_in_L0) {
#if defined(_OPENMP)
#    pragma omp task default(none) \
        shared(hmatrix)            \
        depend(inout : hmatrix)

#endif
            {
                sequential_lu_factorization(hmatrix);
            }
        } else {

            bool block_tree_not_consistent = (hmatrix.get_target_cluster().get_rank() < 0 || hmatrix.get_source_cluster().get_rank() < 0);
            std::vector<const Cluster<CoordinatePrecision> *> clusters;
            const Cluster<CoordinatePrecision> &cluster = hmatrix.get_target_cluster();
            fill_clusters(cluster, block_tree_not_consistent, clusters);

            for (auto &cluster_child : clusters) {
                HMatrix<CoefficientPrecision, CoordinatePrecision> *pivot = hmatrix.get_sub_hmatrix(*cluster_child, *cluster_child);
                // Compute pivot block
                task_based_lu_factorization(*pivot, L0);

                // Apply pivot block to row and column
                for (auto &other_cluster_child : clusters) {
                    if (other_cluster_child->get_offset() > cluster_child->get_offset()) {
                        HMatrix<CoefficientPrecision, CoordinatePrecision> *U = hmatrix.get_sub_hmatrix(*cluster_child, *other_cluster_child);
                        HMatrix<CoefficientPrecision, CoordinatePrecision> *L = hmatrix.get_sub_hmatrix(*other_cluster_child, *cluster_child);

                        task_based_internal_triangular_hmatrix_hmatrix_solve('L', 'L', 'N', 'U', CoefficientPrecision(1), *pivot, *U, L0, L0);
                        task_based_internal_triangular_hmatrix_hmatrix_solve('R', 'U', 'N', 'N', CoefficientPrecision(1), *pivot, *L, L0, L0);
                    }
                }

                // Update Schur complement
                for (auto &output_cluster_child : clusters) {
                    for (auto &input_cluster_child : clusters) {
                        if (output_cluster_child->get_offset() > cluster_child->get_offset() && input_cluster_child->get_offset() > cluster_child->get_offset()) {
                            HMatrix<CoefficientPrecision, CoordinatePrecision> *A_child = hmatrix.get_sub_hmatrix(*output_cluster_child, *input_cluster_child);
                            const HMatrix<CoefficientPrecision, CoordinatePrecision> *U = hmatrix.get_sub_hmatrix(*cluster_child, *input_cluster_child);
                            const HMatrix<CoefficientPrecision, CoordinatePrecision> *L = hmatrix.get_sub_hmatrix(*output_cluster_child, *cluster_child);

                            task_based_internal_add_hmatrix_hmatrix_product('N', 'N', CoefficientPrecision(-1), *L, *U, CoefficientPrecision(1), *A_child, L0, L0, L0);
                        }
                    }
                }
            }
        }
    } else if (hmatrix.is_dense()) {
#if defined(_OPENMP)
#    pragma omp task default(none) \
        shared(hmatrix)            \
        depend(inout : hmatrix)

#endif
        {
            lu_factorization(*hmatrix.get_dense_data());
        }
    } else {
        htool::Logger::get_instance().log(LogLevel::ERROR, "Operation is not implemented for task_based_lu_factorization (hmatrix is low-rank)"); // LCOV_EXCL_LINE
    }
} // end of task_based_lu_factorization

/**
 * @brief Task-based Cholesky factorization of a hierarchical matrix.
 *
 * @param UPLO specifies whether the upper or lower triangular part of the matrix is used.
 * @param hmatrix the hierarchical matrix to be factorized.
 * @param L0 A vector of pointers to nodes of the output 'hmatrix'.
 *
 * This function performs a Cholesky factorization of a hierarchical matrix. The factorization is
 * done in a task-based way, i.e. the factorization of the sub-matrices is done in parallel using
 * OpenMP tasks. The function is only implemented for consistent block trees.
 *
 */
template <typename CoefficientPrecision, typename CoordinatePrecision = underlying_type<CoefficientPrecision>>
void task_based_cholesky_factorization(char UPLO, HMatrix<CoefficientPrecision, CoordinatePrecision> &hmatrix, std::vector<HMatrix<CoefficientPrecision> *> &L0) {
    if (hmatrix.is_hierarchical()) {
        // check if hmatrix is in L0.
        bool is_hmatrix_in_L0 = false;
        for (auto &L0_node : L0) {
            if (L0_node->get_target_cluster() == hmatrix.get_target_cluster() && L0_node->get_source_cluster() == hmatrix.get_source_cluster()) {
                is_hmatrix_in_L0 = true;
                break;
            }
        }

        if (is_hmatrix_in_L0) {
#if defined(_OPENMP)
#    pragma omp task default(none) \
        firstprivate(UPLO)         \
        shared(hmatrix)            \
        depend(inout : hmatrix)
#endif
            {
                sequential_cholesky_factorization(UPLO, hmatrix);
            }
        } else {

            bool block_tree_not_consistent = (hmatrix.get_target_cluster().get_rank() < 0 || hmatrix.get_source_cluster().get_rank() < 0);
            std::vector<const Cluster<CoordinatePrecision> *> clusters;
            const Cluster<CoordinatePrecision> &cluster = hmatrix.get_target_cluster();
            fill_clusters(cluster, block_tree_not_consistent, clusters);

            for (auto &cluster_child : clusters) {
                HMatrix<CoefficientPrecision, CoordinatePrecision> *pivot = hmatrix.get_sub_hmatrix(*cluster_child, *cluster_child);
                // Compute pivot block
                task_based_cholesky_factorization(UPLO, *pivot, L0);

                // Apply pivot block to row and column
                for (auto &other_cluster_child : clusters) {
                    if (other_cluster_child->get_offset() > cluster_child->get_offset()) {
                        if (UPLO == 'L') {
                            HMatrix<CoefficientPrecision, CoordinatePrecision> *L = hmatrix.get_sub_hmatrix(*other_cluster_child, *cluster_child);
                            task_based_internal_triangular_hmatrix_hmatrix_solve('R', UPLO, is_complex<CoefficientPrecision>() ? 'C' : 'T', 'N', CoefficientPrecision(1), *pivot, *L, L0, L0);

                        } else {
                            HMatrix<CoefficientPrecision, CoordinatePrecision> *U = hmatrix.get_sub_hmatrix(*cluster_child, *other_cluster_child);
                            task_based_internal_triangular_hmatrix_hmatrix_solve('L', UPLO, is_complex<CoefficientPrecision>() ? 'C' : 'T', 'N', CoefficientPrecision(1), *pivot, *U, L0, L0);
                        }
                    }
                }

                // Update Schur complement
                for (auto &output_cluster_child : clusters) {
                    for (auto &input_cluster_child : clusters) {
                        if (UPLO == 'L' && output_cluster_child->get_offset() > cluster_child->get_offset() && input_cluster_child->get_offset() > cluster_child->get_offset() && output_cluster_child->get_offset() >= input_cluster_child->get_offset()) {
                            HMatrix<CoefficientPrecision, CoordinatePrecision> *A_child = hmatrix.get_sub_hmatrix(*output_cluster_child, *input_cluster_child);
                            const HMatrix<CoefficientPrecision, CoordinatePrecision> *L = hmatrix.get_sub_hmatrix(*output_cluster_child, *cluster_child);
                            task_based_internal_add_hmatrix_hmatrix_product('N', is_complex<CoefficientPrecision>() ? 'C' : 'T', CoefficientPrecision(-1), *L, *L, CoefficientPrecision(1), *A_child, L0, L0, L0);

                        } else if (UPLO == 'U' && output_cluster_child->get_offset() > cluster_child->get_offset() && input_cluster_child->get_offset() > cluster_child->get_offset() && input_cluster_child->get_offset() >= output_cluster_child->get_offset()) {
                            HMatrix<CoefficientPrecision, CoordinatePrecision> *A_child = hmatrix.get_sub_hmatrix(*output_cluster_child, *input_cluster_child);
                            const HMatrix<CoefficientPrecision, CoordinatePrecision> *U = hmatrix.get_sub_hmatrix(*cluster_child, *input_cluster_child);
                            task_based_internal_add_hmatrix_hmatrix_product(is_complex<CoefficientPrecision>() ? 'C' : 'T', 'N', CoefficientPrecision(-1), *U, *U, CoefficientPrecision(1), *A_child, L0, L0, L0);
                        }
                    }
                }
            }
        }
    } else if (hmatrix.is_dense()) {
#if defined(_OPENMP)
#    pragma omp task default(none) \
        firstprivate(UPLO)         \
        shared(hmatrix)            \
        depend(inout : hmatrix)

#endif
        {
            cholesky_factorization(UPLO, *hmatrix.get_dense_data());
        }
    } else {
        htool::Logger::get_instance().log(LogLevel::ERROR, "Operation is not implemented for task_based_cholesky_factorization (hmatrix is low-rank)"); // LCOV_EXCL_LINE
    }
} // end of task_based_cholesky_factorization

// Copies the block tree of hmatrix into copy, which has the same clusters, without the data of its leaves.
template <typename CoefficientPrecision, typename CoordinatePrecision>
void copy_block_tree(const HMatrix<CoefficientPrecision, CoordinatePrecision> &hmatrix, HMatrix<CoefficientPrecision, CoordinatePrecision> &copy) {
    copy.set_symmetry(hmatrix.get_symmetry());
    copy.set_UPLO(hmatrix.get_UPLO());
    copy.set_symmetry_for_leaves(hmatrix.get_symmetry_for_leaves());
    copy.set_UPLO_for_leaves(hmatrix.get_UPLO_for_leaves());
    for (auto &child : hmatrix.get_children()) {
        copy_block_tree(*child, *copy.add_child(&child->get_target_cluster(), &child->get_source_cluster()));
    }
}

// Copies the data of the leaves of hmatrix into copy, which has the same block tree (see copy_block_tree).
template <typename CoefficientPrecision, typename CoordinatePrecision>
void copy_leaves_data(const HMatrix<CoefficientPrecision, CoordinatePrecision> &hmatrix, HMatrix<CoefficientPrecision, CoordinatePrecision> &copy) {
    if (hmatrix.is_dense()) {
        copy.set_dense_data(std::make_unique<Matrix<CoefficientPrecision>>(*hmatrix.get_dense_data()));
    } else if (hmatrix.is_low_rank()) {
        copy.set_low_rank_data(std::make_unique<LowRankMatrix<CoefficientPrecision>>(*hmatrix.get_low_rank_data()));
    } else {
        for (std::size_t c = 0; c < hmatrix.get_children().size(); c++) {
            copy_leaves_data(*hmatrix.get_children()[c], *copy.get_children()[c]);
        }
    }
}

/**
 * @brief Task-based LDLt factorization of a hierarchical matrix, following sequential_ldlt_factorization.
 *
 * @param symmetry 'S' for symmetric, 'H' for Hermitian.
 * @param UPLO specifies whether the upper or lower triangular part of the matrix is used.
 * @param hmatrix the hierarchical matrix to be factorized.
 * @param L0 A vector of pointers to nodes of the output 'hmatrix'.
 */
template <typename CoefficientPrecision, typename CoordinatePrecision = underlying_type<CoefficientPrecision>>
void task_based_ldlt_factorization(char symmetry, char UPLO, HMatrix<CoefficientPrecision, CoordinatePrecision> &hmatrix, std::vector<HMatrix<CoefficientPrecision> *> &L0) {
    char transa = symmetry == 'H' ? 'C' : 'T';
    if (hmatrix.is_hierarchical()) {
        // check if hmatrix is in L0.
        bool is_hmatrix_in_L0 = false;
        for (auto &L0_node : L0) {
            if (L0_node->get_target_cluster() == hmatrix.get_target_cluster() && L0_node->get_source_cluster() == hmatrix.get_source_cluster()) {
                is_hmatrix_in_L0 = true;
                break;
            }
        }

        if (is_hmatrix_in_L0) {
#if defined(_OPENMP)
#    pragma omp task default(none)   \
        firstprivate(symmetry, UPLO) \
        shared(hmatrix)              \
        depend(inout : hmatrix)
#endif
            {
                sequential_ldlt_factorization(symmetry, UPLO, hmatrix);
            }
        } else {
            bool block_tree_not_consistent = (hmatrix.get_target_cluster().get_rank() < 0 || hmatrix.get_source_cluster().get_rank() < 0);
            std::vector<const Cluster<CoordinatePrecision> *> clusters;
            fill_clusters(hmatrix.get_target_cluster(), block_tree_not_consistent, clusters);
            if (UPLO == 'U') {
                std::reverse(clusters.begin(), clusters.end());
            }

            for (std::size_t p = 0; p < clusters.size(); p++) {
                HMatrix<CoefficientPrecision, CoordinatePrecision> *pivot = hmatrix.get_sub_hmatrix(*clusters[p], *clusters[p]);
                task_based_ldlt_factorization(symmetry, UPLO, *pivot, L0);

                // W(q) = A(q,p) L(p,p)^-T is copied, the stored block becoming L(q,p) = W(q) D(p)^-1.
                // The copies are created with their block tree, so that tasks using them can be created now, and their data is copied by tasks.
                auto Ws = std::make_shared<std::vector<std::unique_ptr<HMatrix<CoefficientPrecision, CoordinatePrecision>>>>();
                std::vector<HMatrix<CoefficientPrecision, CoordinatePrecision> *> Ls;
                std::vector<HMatrix<CoefficientPrecision, CoordinatePrecision> *> Ls_L0_nodes;
                for (std::size_t q = p + 1; q < clusters.size(); q++) {
                    HMatrix<CoefficientPrecision, CoordinatePrecision> *L = hmatrix.get_sub_hmatrix(*clusters[q], *clusters[p]);
                    task_based_internal_triangular_ldlt_hmatrix_hmatrix_solve('R', UPLO, transa, CoefficientPrecision(1), *pivot, *L, L0, L0);

                    auto W = std::make_unique<HMatrix<CoefficientPrecision, CoordinatePrecision>>(L->get_target_cluster(), L->get_source_cluster());
                    W->set_epsilon(L->get_epsilon());
                    W->set_block_tree_consistency(L->is_block_tree_consistent());
                    copy_block_tree(*L, *W);
                    for (HMatrix<CoefficientPrecision, CoordinatePrecision> *L0_node : enumerate_dependences(*L, L0)) {
                        HMatrix<CoefficientPrecision, CoordinatePrecision> *W_node = W->get_sub_hmatrix(L0_node->get_target_cluster(), L0_node->get_source_cluster());
                        Ls_L0_nodes.push_back(L0_node);
#if defined(_OPENMP)
#    pragma omp task default(none)    \
        firstprivate(L0_node, W_node) \
        depend(in : *L0_node)
#endif
                        {
                            copy_leaves_data(*L0_node, *W_node);
                        }
                    }

                    task_based_internal_apply_ldlt_diagonal(symmetry, 'R', UPLO, *pivot, *L, L0, L0);
                    Ls.push_back(L);
                    Ws->push_back(std::move(W));
                }

                // Schur complement on the stored triangle: A(q,r) -= L(q,p) D(p) L(r,p)^T = L(q,p) W(r)^T
                for (std::size_t q = 0; q < Ls.size(); q++) {
                    for (std::size_t r = 0; r < q + 1; r++) {
                        HMatrix<CoefficientPrecision, CoordinatePrecision> *A_child = hmatrix.get_sub_hmatrix(*clusters[p + 1 + q], *clusters[p + 1 + r]);
                        task_based_internal_add_hmatrix_hmatrix_product('N', transa, CoefficientPrecision(-1), *Ls[q], *(*Ws)[r], CoefficientPrecision(1), *A_child, L0, L0, L0);
                    }
                }

                // The copies are freed once the products using them, which depend on the L0 nodes of the blocks L(r,p), are completed
                if (!Ls_L0_nodes.empty()) {
#if defined(_OPENMP)
                    int Ls_L0_nodes_size = Ls_L0_nodes.size();
#    pragma omp task default(none) \
        firstprivate(Ws)           \
        shared(Ls_L0_nodes)        \
        depend(iterator(it = 0 : Ls_L0_nodes_size), inout : *Ls_L0_nodes[it])
#endif
                    {
                        Ws->clear();
                    }
                }
            }
        }
    } else if (hmatrix.is_dense()) {
#if defined(_OPENMP)
#    pragma omp task default(none)   \
        firstprivate(symmetry, UPLO) \
        shared(hmatrix)              \
        depend(inout : hmatrix)
#endif
        {
            ldlt_factorization(symmetry, UPLO, *hmatrix.get_dense_data());
        }
    } else {
        htool::Logger::get_instance().log(LogLevel::ERROR, "Operation is not implemented for task_based_ldlt_factorization (hmatrix is low-rank)"); // LCOV_EXCL_LINE
    }
} // end of task_based_ldlt_factorization

} // namespace htool

#if defined(__clang__)
#elif defined(__GNUC__) || defined(__GNUG__)
#    pragma GCC diagnostic pop
#endif

#endif
