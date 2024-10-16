#ifndef HTOOL_HMATRIX_LINALG_ADD_HMATRIX_HMATRIX_PRODUCT_HPP
#define HTOOL_HMATRIX_LINALG_ADD_HMATRIX_HMATRIX_PRODUCT_HPP

#include "../../clustering/cluster_node.hpp"
#include "../../matrix/matrix.hpp"
#include "../hmatrix.hpp"
#include "../lrmat/linalg/add_lrmat_lrmat.hpp"
#include "../lrmat/linalg/add_lrmat_lrmat_product.hpp"
#include "../lrmat/linalg/add_lrmat_matrix_product.hpp"
#include "../lrmat/linalg/add_matrix_lrmat_product.hpp"
#include "../lrmat/linalg/add_matrix_matrix_product.hpp"
#include "../lrmat/lrmat.hpp"
#include "add_hmatrix_lrmat_product.hpp"
#include "add_hmatrix_matrix_product.hpp"
#include "add_lrmat_hmatrix.hpp"
#include "add_lrmat_hmatrix_product.hpp"
#include "add_matrix_hmatrix_product.hpp"
#include "scale.hpp"
#include <vector>

namespace htool {

template <typename CoefficientPrecision, typename CoordinatePrecision = CoefficientPrecision>
void internal_add_hmatrix_hmatrix_product(char transa, char transb, CoefficientPrecision alpha, const HMatrix<CoefficientPrecision, CoordinatePrecision> &A, const HMatrix<CoefficientPrecision, CoordinatePrecision> &B, CoefficientPrecision beta, LowRankMatrix<CoefficientPrecision> &C) {
    if (beta != CoefficientPrecision(1)) {
        scale(beta, C);
    }

    if (A.is_hierarchical() and B.is_hierarchical()) {

        bool block_tree_not_consistent = (A.get_target_cluster().get_rank() < 0 || A.get_source_cluster().get_rank() < 0 || B.get_target_cluster().get_rank() < 0 || B.get_source_cluster().get_rank() < 0);

        std::vector<LowRankMatrix<CoefficientPrecision>> low_rank_matrices;
        std::vector<const Cluster<CoordinatePrecision> *> output_clusters, middle_clusters, input_clusters;

        const Cluster<CoordinatePrecision> &output_cluster = transa == 'N' ? A.get_target_cluster() : A.get_source_cluster();
        const Cluster<CoordinatePrecision> &middle_cluster = transa == 'N' ? A.get_source_cluster() : A.get_target_cluster();
        const Cluster<CoordinatePrecision> &input_cluster  = transb == 'N' ? B.get_source_cluster() : B.get_target_cluster();

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

        for (auto &output_cluster_child : output_clusters) {
            for (auto &input_cluster_child : input_clusters) {
                low_rank_matrices.emplace_back(C.get_epsilon());
                low_rank_matrices.back().get_U().resize(output_cluster_child->get_size(), 0);
                low_rank_matrices.back().get_V().resize(0, input_cluster_child->get_size());
                for (auto &middle_cluster_child : middle_clusters) {
                    const HMatrix<CoefficientPrecision, CoordinatePrecision> *A_child = (transa == 'N') ? A.get_sub_hmatrix(*output_cluster_child, *middle_cluster_child) : A.get_sub_hmatrix(*middle_cluster_child, *output_cluster_child);
                    const HMatrix<CoefficientPrecision, CoordinatePrecision> *B_child = (transb == 'N') ? B.get_sub_hmatrix(*middle_cluster_child, *input_cluster_child) : B.get_sub_hmatrix(*input_cluster_child, *middle_cluster_child);
                    internal_add_hmatrix_hmatrix_product(transa, transb, alpha, *A_child, *B_child, CoefficientPrecision(1), low_rank_matrices.back());
                }
            }
        }
        int index = 0;
        for (auto &output_cluster_child : output_clusters) {
            for (auto &input_cluster_child : input_clusters) {
                add_lrmat_lrmat(low_rank_matrices[index], *output_cluster_child, *input_cluster_child, C, output_cluster, input_cluster);
                index++;
            }
        }
    } else {
        if (A.is_dense()) {
            if (B.is_low_rank()) {
                add_matrix_lrmat_product(transa, transb, alpha, *A.get_dense_data(), *B.get_low_rank_data(), CoefficientPrecision(1), C);
            } else if (B.is_dense()) {
                add_matrix_matrix_product(transa, transb, alpha, *A.get_dense_data(), *B.get_dense_data(), CoefficientPrecision(1), C);
            } else {
                internal_add_matrix_hmatrix_product(transa, transb, alpha, *A.get_dense_data(), B, CoefficientPrecision(1), C);
            }
        } else if (A.is_low_rank()) {
            if (B.is_dense()) {
                add_lrmat_matrix_product(transa, transb, alpha, *A.get_low_rank_data(), *B.get_dense_data(), CoefficientPrecision(1), C);
            } else if (B.is_low_rank()) {
                add_lrmat_lrmat_product(transa, transb, alpha, *A.get_low_rank_data(), *B.get_low_rank_data(), CoefficientPrecision(1), C);
            } else {
                internal_add_lrmat_hmatrix_product(transa, transb, alpha, *A.get_low_rank_data(), B, CoefficientPrecision(1), C);
            }
        } else {
            if (B.is_low_rank()) {
                internal_add_hmatrix_lrmat_product(transa, transb, alpha, A, *B.get_low_rank_data(), CoefficientPrecision(1), C);
            } else if (B.is_dense()) {
                internal_add_hmatrix_matrix_product(transa, transb, alpha, A, *B.get_dense_data(), CoefficientPrecision(1), C);
            }
        }
    }
}

template <typename CoefficientPrecision, typename CoordinatePrecision = CoefficientPrecision>
void internal_add_hmatrix_hmatrix_product(char transa, char transb, CoefficientPrecision alpha, const HMatrix<CoefficientPrecision, CoordinatePrecision> &A, const HMatrix<CoefficientPrecision, CoordinatePrecision> &B, CoefficientPrecision beta, Matrix<CoefficientPrecision> &C) {

    if (beta != CoefficientPrecision(1)) {
        scale(beta, C);
    }

    if (A.get_target_cluster().get_rank() < 0 || A.get_source_cluster().get_rank() < 0 || B.get_target_cluster().get_rank() < 0 || B.get_source_cluster().get_rank() < 0) {

        Matrix<CoefficientPrecision> B_dense(B.nb_rows(), B.nb_cols());
        copy_to_dense(B, B_dense.data());
        internal_add_hmatrix_matrix_product(transa, transb, alpha, A, B_dense, CoefficientPrecision(1), C); // It could be optimized, but it should not happen often...
        // std::vector<const Cluster<CoordinatePrecision> *> output_clusters, middle_clusters, input_clusters;
        // auto get_output_cluster_A{&HMatrix<CoefficientPrecision, CoordinatePrecision>::get_target_cluster};
        // auto get_input_cluster_A{&HMatrix<CoefficientPrecision, CoordinatePrecision>::get_source_cluster};

        // if (transa != 'N') {
        //     get_input_cluster_A  = &HMatrix<CoefficientPrecision, CoordinatePrecision>::get_target_cluster;
        //     get_output_cluster_A = &HMatrix<CoefficientPrecision, CoordinatePrecision>::get_source_cluster;
        // }
        // const Cluster<CoordinatePrecision> &output_cluster = (A.*get_output_cluster_A)();
        // const Cluster<CoordinatePrecision> &middle_cluster = (A.*get_input_cluster_A)();
        // const Cluster<CoordinatePrecision> &input_cluster  = B.get_source_cluster();

        // if (output_cluster.get_rank() >= 0) {
        //     output_clusters.push_back(&output_cluster);
        // } else {
        //     for (auto &output_cluster_child : output_cluster.get_clusters_on_partition()) {
        //         output_clusters.push_back(output_cluster_child);
        //     }
        // }
        // if (input_cluster.get_rank() >= 0) {
        //     input_clusters.push_back(&input_cluster);
        // } else {
        //     for (auto &input_cluster_child : input_cluster.get_clusters_on_partition()) {
        //         input_clusters.push_back(input_cluster_child);
        //     }
        // }

        // if (middle_cluster.get_rank() >= 0) {
        //     middle_clusters.push_back(&middle_cluster);
        // } else {
        //     for (auto &middle_cluster_child : middle_cluster.get_clusters_on_partition()) {
        //         middle_clusters.push_back(middle_cluster_child);
        //     }
        // }

        // for (auto &output_cluster_child : output_clusters) {
        //     for (auto &input_cluster_child : input_clusters) {
        //         for (auto &middle_cluster_child : middle_clusters) {
        //             const HMatrix<CoefficientPrecision, CoordinatePrecision> *A_child = (transa == 'N') ? A.get_sub_hmatrix(*output_cluster_child, *middle_cluster_child) : A.get_sub_hmatrix(*middle_cluster_child, *output_cluster_child);
        //             const HMatrix<CoefficientPrecision, CoordinatePrecision> *B_child = B.get_sub_hmatrix(*middle_cluster_child, *input_cluster_child);
        //             internal_add_hmatrix_hmatrix_product(transa, transb, alpha, *A_child, *B_child, CoefficientPrecision(1), C);
        //         }
        //     }
        // }
    } else {
        if (A.is_dense()) {
            if (B.is_low_rank()) {
                add_matrix_lrmat_product(transa, transb, alpha, *A.get_dense_data(), *B.get_low_rank_data(), CoefficientPrecision(1), C);
            } else if (B.is_dense()) {
                add_matrix_matrix_product(transa, transb, alpha, *A.get_dense_data(), *B.get_dense_data(), CoefficientPrecision(1), C);
            } else {
                internal_add_matrix_hmatrix_product(transa, transb, alpha, *A.get_dense_data(), B, CoefficientPrecision(1), C);
            }
        } else if (A.is_low_rank()) {
            if (B.is_dense()) {
                add_lrmat_matrix_product(transa, transb, alpha, *A.get_low_rank_data(), *B.get_dense_data(), CoefficientPrecision(1), C);
            } else if (B.is_low_rank()) {
                add_lrmat_lrmat_product(transa, transb, alpha, *A.get_low_rank_data(), *B.get_low_rank_data(), CoefficientPrecision(1), C);
            } else {
                internal_add_lrmat_hmatrix_product(transa, transb, alpha, *A.get_low_rank_data(), B, CoefficientPrecision(1), C);
            }
        } else {
            if (B.is_low_rank()) {
                internal_add_hmatrix_lrmat_product(transa, transb, alpha, A, *B.get_low_rank_data(), CoefficientPrecision(1), C);
            } else if (B.is_dense()) {
                internal_add_hmatrix_matrix_product(transa, transb, alpha, A, *B.get_dense_data(), CoefficientPrecision(1), C);
            } else {
                Matrix<CoefficientPrecision> B_dense(B.get_target_cluster().get_size(), B.get_source_cluster().get_size());
                copy_to_dense(B, B_dense.data());
                internal_add_hmatrix_matrix_product(transa, transb, alpha, A, B_dense, CoefficientPrecision(1), C);
            }
        }
    }
}

template <typename CoefficientPrecision, typename CoordinatePrecision = CoefficientPrecision>
void internal_add_hmatrix_hmatrix_product(char transa, char transb, CoefficientPrecision alpha, const HMatrix<CoefficientPrecision, CoordinatePrecision> &A, const HMatrix<CoefficientPrecision, CoordinatePrecision> &B, CoefficientPrecision beta, HMatrix<CoefficientPrecision, CoordinatePrecision> &C) {

    if (C.is_dense()) {
        internal_add_hmatrix_hmatrix_product(transa, transb, alpha, A, B, beta, *C.get_dense_data());
    } else if (C.is_low_rank()) {
        internal_add_hmatrix_hmatrix_product(transa, transb, alpha, A, B, beta, *C.get_low_rank_data());
    } else {
        if (A.is_hierarchical() and B.is_hierarchical()) {
            if (beta != CoefficientPrecision(1)) {
                sequential_scale(beta, C);
            }
            bool block_tree_not_consistent = (A.get_target_cluster().get_rank() < 0 || A.get_source_cluster().get_rank() < 0 || B.get_target_cluster().get_rank() < 0 || B.get_source_cluster().get_rank() < 0);

            std::vector<const Cluster<CoordinatePrecision> *> output_clusters, middle_clusters, input_clusters;

            const Cluster<CoordinatePrecision> &output_cluster = transa == 'N' ? A.get_target_cluster() : A.get_source_cluster();
            const Cluster<CoordinatePrecision> &middle_cluster = transa == 'N' ? A.get_source_cluster() : A.get_target_cluster();
            const Cluster<CoordinatePrecision> &input_cluster  = transb == 'N' ? B.get_source_cluster() : B.get_target_cluster();

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

            for (auto &output_cluster_child : output_clusters) {
                for (auto &input_cluster_child : input_clusters) {
                    for (auto &middle_cluster_child : middle_clusters) {
                        auto &&A_child = (transa == 'N') ? A.get_child_or_this(*output_cluster_child, *middle_cluster_child) : A.get_child_or_this(*middle_cluster_child, *output_cluster_child);
                        auto &&B_child = (transb == 'N') ? B.get_child_or_this(*middle_cluster_child, *input_cluster_child) : B.get_child_or_this(*input_cluster_child, *middle_cluster_child);
                        auto &&C_child = C.get_child_or_this(*output_cluster_child, *input_cluster_child);
                        if (A_child != nullptr && B_child != nullptr && C_child != nullptr) {
                            internal_add_hmatrix_hmatrix_product(transa, transb, alpha, *A_child, *B_child, CoefficientPrecision(1), *C.get_child_or_this(*output_cluster_child, *input_cluster_child));
                        }
                    }
                }
            }
        } else {
            if (A.is_low_rank() || B.is_low_rank()) {
                LowRankMatrix<CoefficientPrecision> lrmat(A.is_low_rank() ? A.get_low_rank_data()->get_epsilon() : B.get_low_rank_data()->get_epsilon());

                if (A.is_dense() and B.is_low_rank()) {
                    add_matrix_lrmat_product(transa, transb, alpha, *A.get_dense_data(), *B.get_low_rank_data(), beta, lrmat);
                } else if (A.is_hierarchical() and B.is_low_rank()) {
                    internal_add_hmatrix_lrmat_product(transa, transb, alpha, A, *B.get_low_rank_data(), beta, lrmat);
                } else if (A.is_low_rank() and B.is_low_rank()) {
                    add_lrmat_lrmat_product(transa, transb, alpha, *A.get_low_rank_data(), *B.get_low_rank_data(), beta, lrmat);
                } else if (A.is_low_rank() and B.is_dense()) {
                    add_lrmat_matrix_product(transa, transb, alpha, *A.get_low_rank_data(), *B.get_dense_data(), beta, lrmat);
                } else if (A.is_low_rank() and B.is_hierarchical()) {
                    internal_add_lrmat_hmatrix_product(transa, transb, alpha, *A.get_low_rank_data(), B, beta, lrmat);
                }

                internal_add_lrmat_hmatrix(lrmat, C);
            } else {
                LowRankMatrix<CoefficientPrecision> lrmat(C.get_epsilon());
                if (A.is_dense() and B.is_dense()) {
                    add_matrix_matrix_product(transa, transb, alpha, *A.get_dense_data(), *B.get_dense_data(), beta, lrmat);
                } else if (A.is_dense() and B.is_hierarchical()) {
                    internal_add_matrix_hmatrix_product(transa, transb, alpha, *A.get_dense_data(), B, beta, lrmat);
                } else if (A.is_hierarchical() and B.is_dense()) {
                    internal_add_hmatrix_matrix_product(transa, transb, alpha, A, *B.get_dense_data(), beta, lrmat);
                }

                internal_add_lrmat_hmatrix(lrmat, C);
            }
        }
    }
}

} // namespace htool

#endif
