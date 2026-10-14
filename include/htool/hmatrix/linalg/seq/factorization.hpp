#ifndef HTOOL_HMATRIX_LINALG_SEQ_FACTORIZATION_HPP
#define HTOOL_HMATRIX_LINALG_SEQ_FACTORIZATION_HPP

#include "../../../clustering/cluster_node.hpp"
#include "../../../matrix/linalg/factorization.hpp"
#include "../../../matrix/matrix.hpp"
#include "../../../misc/logger.hpp"
#include "../../../misc/misc.hpp"
#include "../../hmatrix.hpp"
#include "add_hmatrix_hmatrix_product.hpp"
#include "apply_ldlt_diagonal.hpp"
#include "triangular_hmatrix_hmatrix_solve.hpp"
#include "triangular_hmatrix_matrix_solve.hpp"
#include "triangular_ldlt_hmatrix_hmatrix_solve.hpp"
#include <algorithm>
#include <string>
#include <vector>

namespace htool {

template <typename CoefficientPrecision, typename CoordinatePrecision = underlying_type<CoefficientPrecision>>
void sequential_lu_factorization(HMatrix<CoefficientPrecision, CoordinatePrecision> &hmatrix) {
    if (hmatrix.is_hierarchical()) {

        bool block_tree_not_consistent = (hmatrix.get_target_cluster().get_rank() < 0 || hmatrix.get_source_cluster().get_rank() < 0);
        std::vector<const Cluster<CoordinatePrecision> *> clusters;
        const Cluster<CoordinatePrecision> &cluster = hmatrix.get_target_cluster();

        if (cluster.is_leaf() || (block_tree_not_consistent and cluster.get_rank() >= 0)) {
            clusters.push_back(&cluster); // LCOV_EXCL_LINE (hmatrix is a hierarchical diagonal block: its cluster is never a leaf)
        } else if (block_tree_not_consistent) {
            for (auto &output_cluster_child : cluster.get_clusters_on_partition()) {
                clusters.push_back(output_cluster_child);
            }
        } else {
            for (auto &output_cluster_child : cluster.get_children()) {
                clusters.push_back(output_cluster_child.get());
            }
        }

        for (auto &cluster_child : clusters) {
            HMatrix<CoefficientPrecision, CoordinatePrecision> *pivot = hmatrix.get_sub_hmatrix(*cluster_child, *cluster_child);
            // Compute pivot block
            sequential_lu_factorization(*pivot);

            // Apply pivot block to row and column
            for (auto &other_cluster_child : clusters) {
                if (other_cluster_child->get_offset() > cluster_child->get_offset()) {
                    HMatrix<CoefficientPrecision, CoordinatePrecision> *U = hmatrix.get_sub_hmatrix(*cluster_child, *other_cluster_child);
                    HMatrix<CoefficientPrecision, CoordinatePrecision> *L = hmatrix.get_sub_hmatrix(*other_cluster_child, *cluster_child);
                    internal_triangular_hmatrix_hmatrix_solve('L', 'L', 'N', 'U', CoefficientPrecision(1), *pivot, *U);
                    internal_triangular_hmatrix_hmatrix_solve('R', 'U', 'N', 'N', CoefficientPrecision(1), *pivot, *L);
                }
            }

            // Update Schur complement
            for (auto &output_cluster_child : clusters) {
                for (auto &input_cluster_child : clusters) {
                    if (output_cluster_child->get_offset() > cluster_child->get_offset() && input_cluster_child->get_offset() > cluster_child->get_offset()) {
                        HMatrix<CoefficientPrecision, CoordinatePrecision> *A_child = hmatrix.get_sub_hmatrix(*output_cluster_child, *input_cluster_child);
                        const HMatrix<CoefficientPrecision, CoordinatePrecision> *U = hmatrix.get_sub_hmatrix(*cluster_child, *input_cluster_child);
                        const HMatrix<CoefficientPrecision, CoordinatePrecision> *L = hmatrix.get_sub_hmatrix(*output_cluster_child, *cluster_child);

                        internal_add_hmatrix_hmatrix_product('N', 'N', CoefficientPrecision(-1), *L, *U, CoefficientPrecision(1), *A_child);
                    }
                }
            }
        }
    } else if (hmatrix.is_dense()) {
        lu_factorization(*hmatrix.get_dense_data());
    } else {
        htool::Logger::get_instance().log(LogLevel::ERROR, "Operation is not implemented for lu_factorization (hmatrix is low-rank)"); // LCOV_EXCL_LINE
    }
}

template <typename CoefficientPrecision, typename CoordinatePrecision = underlying_type<CoefficientPrecision>>
void sequential_cholesky_factorization(char UPLO, HMatrix<CoefficientPrecision, CoordinatePrecision> &hmatrix) {
    if (hmatrix.is_hierarchical()) {

        bool block_tree_not_consistent = (hmatrix.get_target_cluster().get_rank() < 0 || hmatrix.get_source_cluster().get_rank() < 0);
        std::vector<const Cluster<CoordinatePrecision> *> clusters;
        const Cluster<CoordinatePrecision> &cluster = hmatrix.get_target_cluster();

        if (cluster.is_leaf() || (block_tree_not_consistent and cluster.get_rank() >= 0)) {
            clusters.push_back(&cluster); // LCOV_EXCL_LINE (hmatrix is a hierarchical diagonal block: its cluster is never a leaf)
        } else if (block_tree_not_consistent) {
            for (auto &output_cluster_child : cluster.get_clusters_on_partition()) {
                clusters.push_back(output_cluster_child);
            }
        } else {
            for (auto &output_cluster_child : cluster.get_children()) {
                clusters.push_back(output_cluster_child.get());
            }
        }

        for (auto &cluster_child : clusters) {
            HMatrix<CoefficientPrecision, CoordinatePrecision> *pivot = hmatrix.get_sub_hmatrix(*cluster_child, *cluster_child);
            // Compute pivot block
            sequential_cholesky_factorization(UPLO, *pivot);

            // Apply pivot block to row and column
            for (auto &other_cluster_child : clusters) {
                if (other_cluster_child->get_offset() > cluster_child->get_offset()) {
                    if (UPLO == 'L') {
                        HMatrix<CoefficientPrecision, CoordinatePrecision> *L = hmatrix.get_sub_hmatrix(*other_cluster_child, *cluster_child);
                        internal_triangular_hmatrix_hmatrix_solve('R', UPLO, is_complex<CoefficientPrecision>() ? 'C' : 'T', 'N', CoefficientPrecision(1), *pivot, *L);
                    } else {
                        HMatrix<CoefficientPrecision, CoordinatePrecision> *U = hmatrix.get_sub_hmatrix(*cluster_child, *other_cluster_child);
                        internal_triangular_hmatrix_hmatrix_solve('L', UPLO, is_complex<CoefficientPrecision>() ? 'C' : 'T', 'N', CoefficientPrecision(1), *pivot, *U);
                    }
                }
            }

            // Update Schur complement
            for (auto &output_cluster_child : clusters) {
                for (auto &input_cluster_child : clusters) {
                    if (UPLO == 'L' && output_cluster_child->get_offset() > cluster_child->get_offset() && input_cluster_child->get_offset() > cluster_child->get_offset() && output_cluster_child->get_offset() >= input_cluster_child->get_offset()) {
                        HMatrix<CoefficientPrecision, CoordinatePrecision> *A_child = hmatrix.get_sub_hmatrix(*output_cluster_child, *input_cluster_child);
                        const HMatrix<CoefficientPrecision, CoordinatePrecision> *L = hmatrix.get_sub_hmatrix(*output_cluster_child, *cluster_child);

                        // if (*output_cluster_child == *input_cluster_child) {
                        //     symmetric_rank_k_update(UPLO, 'N', CoefficientPrecision(-1), *L, CoefficientPrecision(1), *A_child);
                        // } else {
                        internal_add_hmatrix_hmatrix_product('N', is_complex<CoefficientPrecision>() ? 'C' : 'T', CoefficientPrecision(-1), *L, *L, CoefficientPrecision(1), *A_child);
                        // }
                    } else if (UPLO == 'U' && output_cluster_child->get_offset() > cluster_child->get_offset() && input_cluster_child->get_offset() > cluster_child->get_offset() && input_cluster_child->get_offset() >= output_cluster_child->get_offset()) {
                        HMatrix<CoefficientPrecision, CoordinatePrecision> *A_child = hmatrix.get_sub_hmatrix(*output_cluster_child, *input_cluster_child);
                        const HMatrix<CoefficientPrecision, CoordinatePrecision> *U = hmatrix.get_sub_hmatrix(*cluster_child, *input_cluster_child);
                        // if (*output_cluster_child == *input_cluster_child) {
                        //     symmetric_rank_k_update(UPLO, 'C', CoefficientPrecision(-1), *U, CoefficientPrecision(1), *A_child);
                        // } else {
                        internal_add_hmatrix_hmatrix_product(is_complex<CoefficientPrecision>() ? 'C' : 'T', 'N', CoefficientPrecision(-1), *U, *U, CoefficientPrecision(1), *A_child);
                        // }
                    }
                }
            }
        }
    } else if (hmatrix.is_dense()) {
        cholesky_factorization(UPLO, *hmatrix.get_dense_data());
    } else {
        htool::Logger::get_instance().log(LogLevel::ERROR, "Operation is not implemented for cholesky_factorization (hmatrix is low-rank)"); // LCOV_EXCL_LINE
    }
}

// Block-recursive LDLt, storing the unit L (or U) in place and D at the dense diagonal leaves.
// UPLO='U' eliminates the last cluster first, matching sytrf/hetrf. Bunch-Kaufman pivoting stays
// within each dense leaf.
template <typename CoefficientPrecision, typename CoordinatePrecision = underlying_type<CoefficientPrecision>>
void sequential_ldlt_factorization(char symmetry, char UPLO, HMatrix<CoefficientPrecision, CoordinatePrecision> &hmatrix) {
    char transa = symmetry == 'H' ? 'C' : 'T';
    if (hmatrix.is_hierarchical()) {
        bool block_tree_not_consistent = (hmatrix.get_target_cluster().get_rank() < 0 || hmatrix.get_source_cluster().get_rank() < 0);
        std::vector<const Cluster<CoordinatePrecision> *> clusters;
        const Cluster<CoordinatePrecision> &cluster = hmatrix.get_target_cluster();

        if (cluster.is_leaf() || (block_tree_not_consistent and cluster.get_rank() >= 0)) {
            clusters.push_back(&cluster); // LCOV_EXCL_LINE (hmatrix is a hierarchical diagonal block: its cluster is never a leaf)
        } else if (block_tree_not_consistent) {
            for (auto &output_cluster_child : cluster.get_clusters_on_partition()) {
                clusters.push_back(output_cluster_child);
            }
        } else {
            for (auto &output_cluster_child : cluster.get_children()) {
                clusters.push_back(output_cluster_child.get());
            }
        }
        if (UPLO == 'U') {
            std::reverse(clusters.begin(), clusters.end());
        }

        for (std::size_t p = 0; p < clusters.size(); p++) {
            HMatrix<CoefficientPrecision, CoordinatePrecision> *pivot = hmatrix.get_sub_hmatrix(*clusters[p], *clusters[p]);
            sequential_ldlt_factorization(symmetry, UPLO, *pivot);

            // W(q) = A(q,p) L(p,p)^-T is kept, the stored block becoming L(q,p) = W(q) D(p)^-1
            std::vector<HMatrix<CoefficientPrecision, CoordinatePrecision>> Ws;
            std::vector<HMatrix<CoefficientPrecision, CoordinatePrecision> *> Ls;
            for (std::size_t q = p + 1; q < clusters.size(); q++) {
                HMatrix<CoefficientPrecision, CoordinatePrecision> *L = hmatrix.get_sub_hmatrix(*clusters[q], *clusters[p]);
                internal_triangular_ldlt_hmatrix_hmatrix_solve('R', UPLO, transa, CoefficientPrecision(1), *pivot, *L);
                Ws.push_back(*L);
                internal_apply_ldlt_diagonal(symmetry, 'R', UPLO, *pivot, *L);
                Ls.push_back(L);
            }

            // Schur complement on the stored triangle: A(q,r) -= L(q,p) D(p) L(r,p)^T = L(q,p) W(r)^T
            for (std::size_t q = 0; q < Ls.size(); q++) {
                for (std::size_t r = 0; r < q + 1; r++) {
                    HMatrix<CoefficientPrecision, CoordinatePrecision> *A_child = hmatrix.get_sub_hmatrix(*clusters[p + 1 + q], *clusters[p + 1 + r]);
                    internal_add_hmatrix_hmatrix_product('N', transa, CoefficientPrecision(-1), *Ls[q], Ws[r], CoefficientPrecision(1), *A_child);
                }
            }
        }
    } else if (hmatrix.is_dense()) {
        ldlt_factorization(symmetry, UPLO, *hmatrix.get_dense_data());
    } else {
        htool::Logger::get_instance().log(LogLevel::ERROR, "Operation is not implemented for ldlt_factorization (hmatrix is low-rank)"); // LCOV_EXCL_LINE
    }
}

} // namespace htool

#endif
