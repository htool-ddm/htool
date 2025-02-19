#ifndef HTOOL_HMATRIX_LINALG_ADD_LOW_RANK_MATRIX_HMATRIX_HPP
#define HTOOL_HMATRIX_LINALG_ADD_LOW_RANK_MATRIX_HMATRIX_HPP

#include "../../basic_types/tree.hpp"          // for preorder_tree_traversal
#include "../../matrix/matrix.hpp"             // for Matrix
#include "../../wrappers/wrapper_blas.hpp"     // for Blas
#include "../hmatrix.hpp"                      // for HMatrix
#include "../lrmat/linalg/add_lrmat_lrmat.hpp" // for add_lrmat_lrmat
#include "../lrmat/lrmat.hpp"                  // for LowRankMatrix
#include <algorithm>                           // for copy_n
#include <vector>                              // for vector
namespace htool {

template <typename CoefficientPrecision, typename CoordinatePrecision = underlying_type<CoefficientPrecision>>
void internal_add_lrmat_hmatrix(const LowRankMatrix<CoefficientPrecision> &lrmat, HMatrix<CoefficientPrecision, CoordinatePrecision> &hmatrix) {
    std::vector<HMatrix<CoefficientPrecision, CoordinatePrecision> *> leaves;
    preorder_tree_traversal(hmatrix, [&leaves](HMatrix<CoefficientPrecision, CoordinatePrecision> &current_node) {
        if (!current_node.is_hierarchical()) {
            leaves.push_back(&current_node);
        }
    });
    for (auto &leaf : leaves) {
        auto &target_cluster = leaf->get_target_cluster();
        auto &source_cluster = leaf->get_source_cluster();

        if (leaf->is_dense()) {
            const Matrix<CoefficientPrecision> &U = lrmat.get_U();
            const Matrix<CoefficientPrecision> &V = lrmat.get_V();
            int row_offset                        = target_cluster.get_offset() - hmatrix.get_target_cluster().get_offset();
            int col_offset                        = source_cluster.get_offset() - hmatrix.get_source_cluster().get_offset();

            const CoefficientPrecision *restricted_V_ptr = V.data() + col_offset * V.nb_rows();
            Matrix<CoefficientPrecision> restricted_U(target_cluster.get_size(), lrmat.rank_of());
            for (int i = 0; i < lrmat.rank_of(); i++) {
                std::copy_n(U.data() + U.nb_rows() * i + row_offset, target_cluster.get_size(), restricted_U.data() + restricted_U.nb_rows() * i);
            }

            char transa                = 'N';
            char transb                = 'N';
            int M                      = target_cluster.get_size();
            int N                      = source_cluster.get_size();
            int K                      = lrmat.rank_of();
            int lda                    = M;
            int ldb                    = K;
            int ldc                    = M;
            CoefficientPrecision alpha = 1;
            CoefficientPrecision beta  = 1;
            Blas<CoefficientPrecision>::gemm(&transa, &transb, &M, &N, &K, &alpha, restricted_U.data(), &lda, restricted_V_ptr, &ldb, &beta, leaf->get_dense_data()->data(), &ldc);
        } else { // leaf is low rank matrix
            add_lrmat_lrmat(lrmat, hmatrix.get_target_cluster(), hmatrix.get_source_cluster(), *leaf->get_low_rank_data(), target_cluster, source_cluster);
        }
    }
}

template <typename CoefficientPrecision, typename CoordinatePrecision = CoefficientPrecision>
void internal_add_lrmat_hmatrix(const LowRankMatrix<CoefficientPrecision> &lrmat, const Cluster<CoordinatePrecision> &target_cluster_lrmat, const Cluster<CoordinatePrecision> &source_cluster_lrmat, HMatrix<CoefficientPrecision, CoordinatePrecision> &hmatrix) {
    // Check if hmatrix is larger
    if (target_cluster_lrmat.get_size() < hmatrix.get_target_cluster().get_size() || source_cluster_lrmat.get_size() < hmatrix.get_source_cluster().get_size()) {
        htool::Logger::get_instance().log(LogLevel::ERROR, "LowRankMatrix is larger than HMatrix"); // LCOV_EXCL_LINE
    }

    std::vector<HMatrix<CoefficientPrecision, CoordinatePrecision> *> leaves;
    preorder_tree_traversal(hmatrix, [&leaves](HMatrix<CoefficientPrecision, CoordinatePrecision> &current_node) {
        if (!current_node.is_hierarchical()) {
            leaves.push_back(&current_node);
        }
    });
    for (auto &leaf : leaves) {
        auto &target_cluster = leaf->get_target_cluster();
        auto &source_cluster = leaf->get_source_cluster();

        if (leaf->is_dense()) {
            const Matrix<CoefficientPrecision> &U = lrmat.get_U();
            const Matrix<CoefficientPrecision> &V = lrmat.get_V();
            int row_offset                        = target_cluster.get_offset() - target_cluster_lrmat.get_offset();
            int col_offset                        = source_cluster.get_offset() - source_cluster_lrmat.get_offset();

            const CoefficientPrecision *restricted_V_ptr = V.data() + col_offset * V.nb_rows();
            Matrix<CoefficientPrecision> restricted_U(target_cluster.get_size(), lrmat.rank_of());
            for (int i = 0; i < lrmat.rank_of(); i++) {
                std::copy_n(U.data() + U.nb_rows() * i + row_offset, target_cluster.get_size(), restricted_U.data() + restricted_U.nb_rows() * i);
            }

            char transa                = 'N';
            char transb                = 'N';
            int M                      = target_cluster.get_size();
            int N                      = source_cluster.get_size();
            int K                      = lrmat.rank_of();
            int lda                    = M;
            int ldb                    = K;
            int ldc                    = M;
            CoefficientPrecision alpha = 1;
            CoefficientPrecision beta  = 1;
            Blas<CoefficientPrecision>::gemm(&transa, &transb, &M, &N, &K, &alpha, restricted_U.data(), &lda, restricted_V_ptr, &ldb, &beta, leaf->get_dense_data()->data(), &ldc);
        } else { // leaf is low rank matrix
            add_lrmat_lrmat(lrmat, target_cluster_lrmat, source_cluster_lrmat, *leaf->get_low_rank_data(), target_cluster, source_cluster);
        }
    }
}

} // namespace htool
#endif
