#ifndef HTOOL_HMATRIX_LINALG_TRIANGULAR_HMATRIX_HMATRIX_SOLVE_HPP
#define HTOOL_HMATRIX_LINALG_TRIANGULAR_HMATRIX_HMATRIX_SOLVE_HPP

#include "../../clustering/cluster_node.hpp"
#include "../../matrix/linalg/factorization.hpp"
#include "../../matrix/matrix.hpp"
#include "../../misc/logger.hpp"
#include "../../misc/misc.hpp"
#include "../hmatrix.hpp"
#include "add_hmatrix_hmatrix_product.hpp"
#include "triangular_hmatrix_lrmat_solve.hpp"
#include "triangular_hmatrix_matrix_solve.hpp"
#include <memory>
#include <string>
#include <vector>

namespace htool {
template <typename CoefficientPrecision, typename CoordinatePrecision = underlying_type<CoefficientPrecision>>
void internal_triangular_hmatrix_hmatrix_solve(char side, char UPLO, char transa, char diag, CoefficientPrecision alpha, const HMatrix<CoefficientPrecision, CoordinatePrecision> &A, HMatrix<CoefficientPrecision, CoordinatePrecision> &B) {
    if (alpha != CoefficientPrecision(1)) {
        scale(alpha, B);
    }
    if (A.is_hierarchical() and B.is_hierarchical()) {
        bool block_tree_not_consistent = (A.get_target_cluster().get_rank() < 0 || A.get_source_cluster().get_rank() < 0 || B.get_target_cluster().get_rank() < 0 || B.get_source_cluster().get_rank() < 0);

        std::vector<const Cluster<CoordinatePrecision> *> output_clusters, middle_clusters, input_clusters;
        auto get_output_cluster_A{&HMatrix<CoefficientPrecision, CoordinatePrecision>::get_target_cluster};
        auto get_input_cluster_A{&HMatrix<CoefficientPrecision, CoordinatePrecision>::get_source_cluster};

        if (transa != 'N') {
            get_input_cluster_A  = &HMatrix<CoefficientPrecision, CoordinatePrecision>::get_target_cluster;
            get_output_cluster_A = &HMatrix<CoefficientPrecision, CoordinatePrecision>::get_source_cluster;
        }
        const Cluster<CoordinatePrecision> &output_cluster = side == 'L' ? (A.*get_output_cluster_A)() : B.get_target_cluster();
        const Cluster<CoordinatePrecision> &middle_cluster = side == 'L' ? (A.*get_input_cluster_A)() : (A.*get_output_cluster_A)();
        const Cluster<CoordinatePrecision> &input_cluster  = side == 'L' ? B.get_source_cluster() : (A.*get_input_cluster_A)();

        if (output_cluster.is_leaf() || (block_tree_not_consistent and output_cluster.get_rank() >= 0)) {
            output_clusters.push_back(&output_cluster);
        } else if (block_tree_not_consistent) {
            for (auto &output_cluster_child : output_cluster.get_clusters_on_partition()) {
                output_clusters.push_back(output_cluster_child);
            }
        } else {
            for (auto &output_cluster_child : output_cluster.get_children()) {
                output_clusters.push_back(output_cluster_child.get());
            }
        }

        if (input_cluster.is_leaf() || (block_tree_not_consistent and input_cluster.get_rank() >= 0)) {
            input_clusters.push_back(&input_cluster);
        } else if (block_tree_not_consistent) {
            for (auto &input_cluster_child : input_cluster.get_clusters_on_partition()) {
                input_clusters.push_back(input_cluster_child);
            }
        } else {
            for (auto &input_cluster_child : input_cluster.get_children()) {
                input_clusters.push_back(input_cluster_child.get());
            }
        }

        if (middle_cluster.is_leaf() || (block_tree_not_consistent and middle_cluster.get_rank() >= 0)) {
            middle_clusters.push_back(&middle_cluster);
        } else if (block_tree_not_consistent) {
            for (auto &middle_cluster_child : middle_cluster.get_clusters_on_partition()) {
                middle_clusters.push_back(middle_cluster_child);
            }
        } else {
            for (auto &middle_cluster_child : middle_cluster.get_children()) {
                middle_clusters.push_back(middle_cluster_child.get());
            }
        }

        // Forward, compute each block rows one after the other
        if ((UPLO == 'L' && transa == 'N' && side == 'L') || (UPLO == 'U' && transa != 'N' && side == 'L')) {
            for (auto &output_cluster_child : output_clusters) {
                for (auto &input_cluster_child : input_clusters) {
                    HMatrix<CoefficientPrecision, CoordinatePrecision> *B_child_to_modify = B.get_sub_hmatrix(*output_cluster_child, *input_cluster_child);

                    for (auto &middle_cluster_child : middle_clusters) {
                        const HMatrix<CoefficientPrecision, CoordinatePrecision> *A_child = transa == 'N' ? A.get_sub_hmatrix(*output_cluster_child, *middle_cluster_child) : A.get_sub_hmatrix(*middle_cluster_child, *output_cluster_child);
                        if (*output_cluster_child == *middle_cluster_child) {
                            internal_triangular_hmatrix_hmatrix_solve(side, UPLO, transa, diag, CoefficientPrecision(1), *A_child, *B_child_to_modify);
                        } else if (output_cluster_child->get_offset() > middle_cluster_child->get_offset()) {

                            const HMatrix<CoefficientPrecision, CoordinatePrecision> *B_child = B.get_sub_hmatrix(*middle_cluster_child, *input_cluster_child);

                            internal_add_hmatrix_hmatrix_product(transa, 'N', CoefficientPrecision(-1), *A_child, *B_child, CoefficientPrecision(1), *B_child_to_modify);
                        }
                    }
                }
            }
        } // Backward, compute each block rows one after the other
        else if ((UPLO == 'U' && transa == 'N' && side == 'L') || (UPLO == 'L' && transa != 'N' && side == 'L')) {
            for (auto output_it = output_clusters.rbegin(); output_it != output_clusters.rend(); ++output_it) {
                auto &output_cluster_child = *output_it;

                for (auto input_it = input_clusters.rbegin(); input_it != input_clusters.rend(); ++input_it) {
                    auto &input_cluster_child = *input_it;

                    HMatrix<CoefficientPrecision, CoordinatePrecision> *B_child_to_modify = B.get_sub_hmatrix(*output_cluster_child, *input_cluster_child);

                    for (auto middle_it = middle_clusters.rbegin(); middle_it != middle_clusters.rend(); ++middle_it) {
                        auto &middle_cluster_child = *middle_it;

                        const HMatrix<CoefficientPrecision, CoordinatePrecision> *A_child = transa == 'N' ? A.get_sub_hmatrix(*output_cluster_child, *middle_cluster_child) : A.get_sub_hmatrix(*middle_cluster_child, *output_cluster_child);
                        if (*output_cluster_child == *middle_cluster_child) {
                            internal_triangular_hmatrix_hmatrix_solve(side, UPLO, transa, diag, CoefficientPrecision(1), *A_child, *B_child_to_modify);
                        } else if (output_cluster_child->get_offset() < middle_cluster_child->get_offset()) {

                            const HMatrix<CoefficientPrecision, CoordinatePrecision> *B_child = B.get_sub_hmatrix(*middle_cluster_child, *input_cluster_child);

                            internal_add_hmatrix_hmatrix_product(transa, 'N', CoefficientPrecision(-1), *A_child, *B_child, CoefficientPrecision(1), *B_child_to_modify);
                        }
                    }
                }
            }
        } // Forward, compute each block column one after the other
        else if ((UPLO == 'U' && transa == 'N' && side == 'R') || (UPLO == 'L' && transa != 'N' && side == 'R')) {
            for (auto input_it = input_clusters.begin(); input_it != input_clusters.end(); ++input_it) {
                auto &input_cluster_child = *input_it;

                for (auto output_it = output_clusters.begin(); output_it != output_clusters.end(); ++output_it) {
                    auto &output_cluster_child = *output_it;

                    HMatrix<CoefficientPrecision, CoordinatePrecision> *B_child_to_modify = B.get_sub_hmatrix(*output_cluster_child, *input_cluster_child);

                    for (auto middle_it = middle_clusters.begin(); middle_it != middle_clusters.end(); ++middle_it) {
                        auto &middle_cluster_child = *middle_it;

                        const HMatrix<CoefficientPrecision, CoordinatePrecision> *A_child = transa == 'N' ? A.get_sub_hmatrix(*middle_cluster_child, *input_cluster_child) : A.get_sub_hmatrix(*input_cluster_child, *middle_cluster_child);
                        if (*input_cluster_child == *middle_cluster_child) {
                            internal_triangular_hmatrix_hmatrix_solve(side, UPLO, transa, diag, CoefficientPrecision(1), *A_child, *B_child_to_modify);
                        } else if (middle_cluster_child->get_offset() < input_cluster_child->get_offset()) {

                            const HMatrix<CoefficientPrecision, CoordinatePrecision> *B_child = B.get_sub_hmatrix(*output_cluster_child, *middle_cluster_child);

                            internal_add_hmatrix_hmatrix_product('N', transa, CoefficientPrecision(-1), *B_child, *A_child, CoefficientPrecision(1), *B_child_to_modify);
                        }
                    }
                }
            }
        } // Backward, compute each block column one after the other
        else if ((UPLO == 'L' && transa == 'N' && side == 'R') || (UPLO == 'U' && transa != 'N' && side == 'R')) {
            for (auto input_it = input_clusters.rbegin(); input_it != input_clusters.rend(); ++input_it) {
                auto &input_cluster_child = *input_it;

                for (auto output_it = output_clusters.rbegin(); output_it != output_clusters.rend(); ++output_it) {
                    auto &output_cluster_child = *output_it;

                    HMatrix<CoefficientPrecision, CoordinatePrecision> *B_child_to_modify = B.get_sub_hmatrix(*output_cluster_child, *input_cluster_child);

                    for (auto middle_it = middle_clusters.rbegin(); middle_it != middle_clusters.rend(); ++middle_it) {
                        auto &middle_cluster_child = *middle_it;

                        const HMatrix<CoefficientPrecision, CoordinatePrecision> *A_child = transa == 'N' ? A.get_sub_hmatrix(*middle_cluster_child, *input_cluster_child) : A.get_sub_hmatrix(*input_cluster_child, *middle_cluster_child);
                        if (*input_cluster_child == *middle_cluster_child) {
                            internal_triangular_hmatrix_hmatrix_solve(side, UPLO, transa, diag, CoefficientPrecision(1), *A_child, *B_child_to_modify);
                        } else if (middle_cluster_child->get_offset() > input_cluster_child->get_offset()) {

                            const HMatrix<CoefficientPrecision, CoordinatePrecision> *B_child = B.get_sub_hmatrix(*output_cluster_child, *middle_cluster_child);

                            internal_add_hmatrix_hmatrix_product('N', transa, CoefficientPrecision(-1), *B_child, *A_child, CoefficientPrecision(1), *B_child_to_modify);
                        }
                    }
                }
            }
        }
    } else {
        if (A.is_dense()) {
            if (B.is_dense()) {
                triangular_matrix_matrix_solve(side, UPLO, transa, diag, CoefficientPrecision(1), *A.get_dense_data(), *B.get_dense_data());
            } else if (B.is_low_rank()) {
                if (side == 'L' or side == 'l') {
                    triangular_matrix_matrix_solve('L', UPLO, transa, diag, CoefficientPrecision(1), *A.get_dense_data(), B.get_low_rank_data()->get_U());
                } else {
                    triangular_matrix_matrix_solve('R', UPLO, transa, diag, CoefficientPrecision(1), *A.get_dense_data(), B.get_low_rank_data()->get_V());
                }
            } else {
                std::unique_ptr<Matrix<CoefficientPrecision>> B_dense_ptr = std::make_unique<Matrix<CoefficientPrecision>>(B.nb_rows(), B.nb_cols());
                auto &B_dense                                             = *B_dense_ptr;
                copy_to_dense(B, B_dense.data());
                B.set_dense_data(std::move(B_dense_ptr));
                triangular_matrix_matrix_solve(side, UPLO, transa, diag, CoefficientPrecision(1), *A.get_dense_data(), *B.get_dense_data());
            }
        } else if (A.is_low_rank()) {
            htool::Logger::get_instance().log(LogLevel::ERROR, "In triangular_hmatrix_hmatrix_solve, triangular_low_rank_* would be called, which does not make sense."); // LCOV_EXCL_LINE
        } else {
            if (B.is_low_rank()) {
                internal_triangular_hmatrix_lrmat_solve(side, UPLO, transa, diag, CoefficientPrecision(1), A, *B.get_low_rank_data());
            } else if (B.is_dense()) {
                internal_triangular_hmatrix_matrix_solve(side, UPLO, transa, diag, CoefficientPrecision(1), A, *B.get_dense_data());
            }
        }
    }
}
} // namespace htool
#endif
