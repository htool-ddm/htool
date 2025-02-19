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
#include "task_based_triangular_hmatrix_hmatrix_solve.hpp" // for task_based_internal_triangular_hmatrix_hmatrix_solve
#include <string>                                          // for basi...
#include <vector>                                          // for vector

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
    if (!hmatrix.is_block_tree_consistent()) {
        htool::Logger::get_instance().log(LogLevel::ERROR, "task_based_lu_factorization is only implemented for consistent block tree."); // LCOV_EXCL_LINE
    }

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
#if defined(_OPENMP) && !defined(HTOOL_WITH_PYTHON_INTERFACE)
#    pragma omp task default(none) \
        shared(hmatrix)            \
        depend(inout : hmatrix)

#endif
            {
                lu_factorization(hmatrix);
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
        lu_factorization(*hmatrix.get_dense_data());
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
    if (!hmatrix.is_block_tree_consistent()) {
        htool::Logger::get_instance().log(LogLevel::ERROR, "task_based_cholesky_factorization is only implemented for consistent block tree."); // LCOV_EXCL_LINE
    }
    if ((hmatrix.get_UPLO() != 'S' and !is_complex<CoefficientPrecision>())
        and (hmatrix.get_UPLO() != 'H' and is_complex<CoefficientPrecision>())) {
        htool::Logger::get_instance().log(LogLevel::ERROR, "task_based_cholesky_factorization cannot be used on a HMatrix with UPLO=" + std::string(1, hmatrix.get_UPLO()) + "!=N. You should use another factorization."); // LCOV_EXCL_LINE
    }

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
#if defined(_OPENMP) && !defined(HTOOL_WITH_PYTHON_INTERFACE)
#    pragma omp task default(none) \
        firstprivate(UPLO)         \
        shared(hmatrix)            \
        depend(inout : hmatrix)
#endif
            {
                cholesky_factorization(UPLO, hmatrix);
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
        cholesky_factorization(UPLO, *hmatrix.get_dense_data());
    } else {
        htool::Logger::get_instance().log(LogLevel::ERROR, "Operation is not implemented for task_based_cholesky_factorization (hmatrix is low-rank)"); // LCOV_EXCL_LINE
    }
} // end of task_based_cholesky_factorization

} // namespace htool

#endif
