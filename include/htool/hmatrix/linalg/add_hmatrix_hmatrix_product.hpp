#ifndef HTOOL_HMATRIX_LINALG_ADD_HMATRIX_HMATRIX_PRODUCT_HPP
#define HTOOL_HMATRIX_LINALG_ADD_HMATRIX_HMATRIX_PRODUCT_HPP

#include "../../matrix/linalg/add_matrix_matrix_product.hpp"
#include "../hmatrix.hpp"
#include "../lrmat/linalg/add_lrmat_lrmat.hpp"
#include "scale.hpp"

namespace htool {

template <typename CoefficientPrecision, typename CoordinatePrecision = CoefficientPrecision>
void add_hmatrix_hmatrix_product(char transa, char transb, CoefficientPrecision alpha, const HMatrix<CoefficientPrecision, CoordinatePrecision> &A, const HMatrix<CoefficientPrecision, CoordinatePrecision> &B, CoefficientPrecision beta, LowRankMatrix<CoefficientPrecision, CoordinatePrecision> &C) {
    if (beta != CoefficientPrecision(1)) {
        scale(beta, C);
    }

    if (A.is_hierarchical() and B.is_hierarchical()) {

        bool block_tree_not_consistent = (A.get_target_cluster().get_rank() < 0 || A.get_source_cluster().get_rank() < 0 || B.get_target_cluster().get_rank() < 0 || B.get_source_cluster().get_rank() < 0);

        std::vector<LowRankMatrix<CoefficientPrecision, CoordinatePrecision>>
            low_rank_matrices;
        std::vector<const Cluster<CoordinatePrecision> *> output_clusters, middle_clusters, input_clusters;
        auto get_output_cluster_A{&HMatrix<CoefficientPrecision, CoordinatePrecision>::get_target_cluster};
        auto get_input_cluster_A{&HMatrix<CoefficientPrecision, CoordinatePrecision>::get_source_cluster};

        if (transa != 'N') {
            get_input_cluster_A  = &HMatrix<CoefficientPrecision, CoordinatePrecision>::get_target_cluster;
            get_output_cluster_A = &HMatrix<CoefficientPrecision, CoordinatePrecision>::get_source_cluster;
        }
        const Cluster<CoordinatePrecision> &output_cluster = (A.*get_output_cluster_A)();
        const Cluster<CoordinatePrecision> &middle_cluster = (A.*get_input_cluster_A)();
        const Cluster<CoordinatePrecision> &input_cluster  = B.get_source_cluster();

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
                    const HMatrix<CoefficientPrecision, CoordinatePrecision> *B_child = B.get_sub_hmatrix(*middle_cluster_child, *input_cluster_child);
                    if (A_child->get_symmetry() == 'N') {
                        add_hmatrix_hmatrix_product(transa, transb, alpha, *A_child, *B_child, CoefficientPrecision(1), low_rank_matrices.back());
                    } else {
                        add_hmatrix_hmatrix_product_symmetry('L', transa, transb, alpha, *A_child, *B_child, CoefficientPrecision(1), low_rank_matrices.back(), A_child->get_UPLO(), A_child->get_symmetry());
                    }
                }
            }
        }
        int index = 0;
        for (auto &output_cluster_child : output_clusters) {
            for (auto &input_cluster_child : input_clusters) {
                add_lrmat_lrmat(C, output_cluster, input_cluster, low_rank_matrices[index], *output_cluster_child, *input_cluster_child);
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
                add_matrix_hmatrix_product(transa, transb, alpha, *A.get_dense_data(), B, CoefficientPrecision(1), C);
            }
        } else if (A.is_low_rank()) {
            if (B.is_dense()) {
                add_lrmat_matrix_product(transa, transb, alpha, *A.get_low_rank_data(), *B.get_dense_data(), CoefficientPrecision(1), C);
            } else if (B.is_low_rank()) {
                add_lrmat_lrmat_product(transa, transb, alpha, *A.get_low_rank_data(), *B.get_low_rank_data(), CoefficientPrecision(1), C);
            } else {
                add_lrmat_hmatrix_product(transa, transb, alpha, *A.get_low_rank_data(), B, CoefficientPrecision(1), C);
            }
        } else {
            if (B.is_low_rank()) {
                add_hmatrix_lrmat_product(transa, transb, alpha, A, *B.get_low_rank_data(), CoefficientPrecision(1), C);
            } else if (B.is_dense()) {
                add_hmatrix_matrix_product(transa, transb, alpha, A, *B.get_dense_data(), CoefficientPrecision(1), C);
            }
        }
    }
}

template <typename CoefficientPrecision, typename CoordinatePrecision = CoefficientPrecision>
void add_hmatrix_hmatrix_product(char transa, char transb, CoefficientPrecision alpha, const HMatrix<CoefficientPrecision, CoordinatePrecision> &A, const HMatrix<CoefficientPrecision, CoordinatePrecision> &B, CoefficientPrecision beta, Matrix<CoefficientPrecision> &C) {

    if (beta != CoefficientPrecision(1)) {
        scale(beta, C);
    }

    if (A.get_target_cluster().get_rank() < 0 || A.get_source_cluster().get_rank() < 0 || B.get_target_cluster().get_rank() < 0 || B.get_source_cluster().get_rank() < 0) {

        Matrix<CoefficientPrecision> B_dense(B.nb_rows(), B.nb_cols());
        copy_to_dense(B, B_dense.data());
        add_hmatrix_matrix_product(transa, 'N', alpha, A, B_dense, CoefficientPrecision(1), C); // It could be optimized, but it should not happen often...
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
        //             add_hmatrix_hmatrix_product(transa, transb, alpha, *A_child, *B_child, CoefficientPrecision(1), C);
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
                add_matrix_hmatrix_product(transa, transb, alpha, *A.get_dense_data(), B, CoefficientPrecision(1), C);
            }
        } else if (A.is_low_rank()) {
            if (B.is_dense()) {
                add_lrmat_matrix_product(transa, transb, alpha, *A.get_low_rank_data(), *B.get_dense_data(), CoefficientPrecision(1), C);
            } else if (B.is_low_rank()) {
                add_lrmat_lrmat_product(transa, transb, alpha, *A.get_low_rank_data(), *B.get_low_rank_data(), CoefficientPrecision(1), C);
            } else {
                add_lrmat_hmatrix_product(transa, transb, alpha, *A.get_low_rank_data(), B, CoefficientPrecision(1), C);
            }
        } else {
            if (B.is_low_rank()) {
                add_hmatrix_lrmat_product(transa, transb, alpha, A, *B.get_low_rank_data(), CoefficientPrecision(1), C);
            } else if (B.is_dense()) {
                add_hmatrix_matrix_product(transa, transb, alpha, A, *B.get_dense_data(), CoefficientPrecision(1), C);
            } else {
                Matrix<CoefficientPrecision> B_dense(B.get_target_cluster().get_size(), B.get_source_cluster().get_size());
                copy_to_dense(B, B_dense.data());
                add_hmatrix_matrix_product(transa, transb, alpha, A, B_dense, CoefficientPrecision(1), C);
            }
        }
    }
}

template <typename CoefficientPrecision, typename CoordinatePrecision = CoefficientPrecision>
void add_hmatrix_hmatrix_product(char transa, char transb, CoefficientPrecision alpha, const HMatrix<CoefficientPrecision, CoordinatePrecision> &A, const HMatrix<CoefficientPrecision, CoordinatePrecision> &B, CoefficientPrecision beta, HMatrix<CoefficientPrecision, CoordinatePrecision> &C) {

    if (C.is_dense()) {
        add_hmatrix_hmatrix_product(transa, transb, alpha, A, B, beta, *C.get_dense_data());
    } else if (C.is_low_rank()) {
        add_hmatrix_hmatrix_product(transa, transb, alpha, A, B, beta, *C.get_low_rank_data());
    } else {
        if (A.is_hierarchical() and B.is_hierarchical()) {
            if (beta != CoefficientPrecision(1)) {
                sequential_scale(beta, C);
            }
            bool block_tree_not_consistent = (A.get_target_cluster().get_rank() < 0 || A.get_source_cluster().get_rank() < 0 || B.get_target_cluster().get_rank() < 0 || B.get_source_cluster().get_rank() < 0);

            std::vector<const Cluster<CoordinatePrecision> *> output_clusters, middle_clusters, input_clusters;
            auto get_output_cluster_A{&HMatrix<CoefficientPrecision, CoordinatePrecision>::get_target_cluster};
            auto get_input_cluster_A{&HMatrix<CoefficientPrecision, CoordinatePrecision>::get_source_cluster};

            if (transa != 'N') {
                get_input_cluster_A  = &HMatrix<CoefficientPrecision, CoordinatePrecision>::get_target_cluster;
                get_output_cluster_A = &HMatrix<CoefficientPrecision, CoordinatePrecision>::get_source_cluster;
            }
            const Cluster<CoordinatePrecision> &output_cluster = (A.*get_output_cluster_A)();
            const Cluster<CoordinatePrecision> &middle_cluster = (A.*get_input_cluster_A)();
            const Cluster<CoordinatePrecision> &input_cluster  = B.get_source_cluster();

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
                        if (A_child->get_symmetry() == 'N') {
                            add_hmatrix_hmatrix_product(transa, transb, alpha, *A_child, *B.get_child_or_this(*middle_cluster_child, *input_cluster_child), CoefficientPrecision(1), *C.get_child_or_this(*output_cluster_child, *input_cluster_child));
                        } else {
                            add_hmatrix_hmatrix_product_symmetry('L', transa, transb, alpha, *A_child, *B.get_child_or_this(*middle_cluster_child, *input_cluster_child), CoefficientPrecision(1), *C.get_child_or_this(*output_cluster_child, *input_cluster_child), A_child->get_UPLO(), A_child->get_symmetry());
                        }
                    }
                }
            }
        } else {
            Matrix<CoefficientPrecision> C_dense(C.get_target_cluster().get_size(), C.get_source_cluster().get_size());
            copy_to_dense(C, C_dense.data());
            if (A.is_dense()) {
                if (B.is_dense()) {
                    add_matrix_matrix_product(transa, transb, alpha, *A.get_dense_data(), *B.get_dense_data(), beta, C_dense);
                } else if (B.is_low_rank()) {
                    add_matrix_lrmat_product(transa, transb, alpha, *A.get_dense_data(), *B.get_low_rank_data(), beta, C_dense);
                } else {
                    add_matrix_hmatrix_product(transa, transb, alpha, *A.get_dense_data(), B, beta, C_dense);
                }
            } else if (A.is_low_rank()) {
                if (B.is_dense()) {
                    add_lrmat_matrix_product(transa, transb, alpha, *A.get_low_rank_data(), *B.get_dense_data(), beta, C_dense);
                } else if (B.is_low_rank()) {
                    add_lrmat_lrmat_product(transa, transb, alpha, *A.get_low_rank_data(), *B.get_low_rank_data(), beta, C_dense);
                } else {
                    add_lrmat_hmatrix_product(transa, transb, alpha, *A.get_low_rank_data(), B, beta, C_dense);
                }
            } else if (A.is_hierarchical()) {
                if (B.is_dense()) {
                    add_hmatrix_matrix_product(transa, transb, alpha, A, *B.get_dense_data(), beta, C_dense);
                } else if (B.is_low_rank()) {
                    add_hmatrix_lrmat_product(transa, transb, alpha, A, *B.get_low_rank_data(), beta, C_dense);
                }
            }
            C.set_dense_data(C_dense);
        }
    }
}

template <typename CoefficientPrecision, typename CoordinatePrecision = CoefficientPrecision>
void add_hmatrix_hmatrix_product_symmetry(char side, char transa, char transb, CoefficientPrecision alpha, const HMatrix<CoefficientPrecision, CoordinatePrecision> &A, const HMatrix<CoefficientPrecision, CoordinatePrecision> &B, CoefficientPrecision beta, LowRankMatrix<CoefficientPrecision, CoordinatePrecision> &C, char UPLO, char symm) {
    if (transb != 'N') {
        htool::Logger::get_instance().log(LogLevel::ERROR, "Operation is not implemented for add_hmatrix_hmatrix_product_symmetric (transb=" + std::string(1, transb) + ")"); // LCOV_EXCL_LINE
    }
    if (side != 'L') {
        htool::Logger::get_instance().log(LogLevel::ERROR, "Operation is not implemented for add_hmatrix_hmatrix_product_symmetric (side=" + std::string(1, side) + ")"); // LCOV_EXCL_LINE
    }
    if (beta != CoefficientPrecision(1)) {
        scale(beta, C);
    }

    if (A.is_hierarchical() and B.is_hierarchical()) {
        bool block_tree_not_consistent = (A.get_target_cluster().get_rank() < 0 || A.get_source_cluster().get_rank() < 0 || B.get_target_cluster().get_rank() < 0 || B.get_source_cluster().get_rank() < 0);

        std::vector<LowRankMatrix<CoefficientPrecision, CoordinatePrecision>>
            low_rank_matrices;
        std::vector<const Cluster<CoordinatePrecision> *> output_clusters, middle_clusters, input_clusters;
        auto get_output_cluster_A{&HMatrix<CoefficientPrecision, CoordinatePrecision>::get_target_cluster};
        auto get_input_cluster_A{&HMatrix<CoefficientPrecision, CoordinatePrecision>::get_source_cluster};

        const Cluster<CoordinatePrecision> &output_cluster = (A.*get_output_cluster_A)();
        const Cluster<CoordinatePrecision> &middle_cluster = (A.*get_input_cluster_A)();
        const Cluster<CoordinatePrecision> &input_cluster  = B.get_source_cluster();

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
                    const HMatrix<CoefficientPrecision, CoordinatePrecision> *A_child, *B_child;
                    char actual_transa;
                    if (output_cluster_child == middle_cluster_child) {
                        A_child = (transa == 'N') ? A.get_sub_hmatrix(*output_cluster_child, *middle_cluster_child) : A.get_sub_hmatrix(*middle_cluster_child, *output_cluster_child);
                        B_child = B.get_sub_hmatrix(*middle_cluster_child, *input_cluster_child);
                        add_hmatrix_hmatrix_product_symmetry(side, transa, transb, alpha, *A_child, *B_child, CoefficientPrecision(1), low_rank_matrices.back(), UPLO, symm);
                    } else if (output_cluster_child->get_offset() >= middle_cluster_child->get_offset() + middle_cluster_child->get_size()) {
                        A_child       = (A.get_UPLO() == 'L') ? A.get_sub_hmatrix(*output_cluster_child, *middle_cluster_child) : A.get_sub_hmatrix(*middle_cluster_child, *output_cluster_child);
                        B_child       = B.get_sub_hmatrix(*middle_cluster_child, *input_cluster_child);
                        actual_transa = (A.get_UPLO() == 'L') ? 'N' : 'T';
                        add_hmatrix_hmatrix_product(actual_transa, transb, alpha, *A_child, *B_child, CoefficientPrecision(1), low_rank_matrices.back());
                    } else {
                        A_child       = (A.get_UPLO() == 'U') ? A.get_sub_hmatrix(*output_cluster_child, *middle_cluster_child) : A.get_sub_hmatrix(*middle_cluster_child, *output_cluster_child);
                        B_child       = B.get_sub_hmatrix(*middle_cluster_child, *input_cluster_child);
                        actual_transa = (A.get_UPLO() == 'U') ? 'N' : 'T';
                        add_hmatrix_hmatrix_product(actual_transa, transb, alpha, *A_child, *B_child, CoefficientPrecision(1), low_rank_matrices.back());
                    }
                }
            }
        }
        int index = 0;
        for (auto &output_cluster_child : output_clusters) {
            for (auto &input_cluster_child : input_clusters) {
                add_lrmat_lrmat(C, output_cluster, input_cluster, low_rank_matrices[index], *output_cluster_child, *input_cluster_child);
                index++;
            }
        }
    } else {
        add_hmatrix_hmatrix_product(transa, transb, alpha, A, B, CoefficientPrecision(1), C);
    }
}

template <typename CoefficientPrecision, typename CoordinatePrecision = CoefficientPrecision>
void add_hmatrix_hmatrix_product_symmetry(char side, char transa, char transb, CoefficientPrecision alpha, const HMatrix<CoefficientPrecision, CoordinatePrecision> &A, const HMatrix<CoefficientPrecision, CoordinatePrecision> &B, CoefficientPrecision beta, Matrix<CoefficientPrecision> &C, char, char) {
    if (transb != 'N') {
        htool::Logger::get_instance().log(LogLevel::ERROR, "Operation is not implemented for add_hmatrix_hmatrix_product_symmetric (transb=" + std::string(1, transb) + ")"); // LCOV_EXCL_LINE
    }
    if (side != 'L') {
        htool::Logger::get_instance().log(LogLevel::ERROR, "Operation is not implemented for add_hmatrix_hmatrix_product_symmetric (side=" + std::string(1, side) + ")"); // LCOV_EXCL_LINE
    }
    if (beta != CoefficientPrecision(1)) {
        scale(beta, C);
    }
    if (A.is_hierarchical() and B.is_hierarchical()) {
        Matrix<CoefficientPrecision> B_dense(B.get_target_cluster().get_size(), B.get_source_cluster().get_size());
        copy_to_dense(B, B_dense.data());
        add_hmatrix_matrix_product(transa, transb, alpha, A, B_dense, CoefficientPrecision(1), C);
    } else {
        add_hmatrix_hmatrix_product(transa, transb, alpha, A, B, CoefficientPrecision(1), C);
    }
}

template <typename CoefficientPrecision, typename CoordinatePrecision = CoefficientPrecision>
void add_hmatrix_hmatrix_product_symmetry(char side, char transa, char transb, CoefficientPrecision alpha, const HMatrix<CoefficientPrecision, CoordinatePrecision> &A, const HMatrix<CoefficientPrecision, CoordinatePrecision> &B, CoefficientPrecision beta, HMatrix<CoefficientPrecision, CoordinatePrecision> &C, char UPLO, char symm) {
    if (transb != 'N') {
        htool::Logger::get_instance().log(LogLevel::ERROR, "Operation is not implemented for add_hmatrix_hmatrix_product_symmetric (transb=" + std::string(1, transb) + ")"); // LCOV_EXCL_LINE
    }
    if (side != 'L') {
        htool::Logger::get_instance().log(LogLevel::ERROR, "Operation is not implemented for add_hmatrix_hmatrix_product_symmetric (side=" + std::string(1, side) + ")"); // LCOV_EXCL_LINE
    }
    if (beta != CoefficientPrecision(1)) {
        scale(beta, C);
    }

    if (A.is_hierarchical() and B.is_hierarchical() and C.is_hierarchical()) {
        bool block_tree_not_consistent = (A.get_target_cluster().get_rank() < 0 || A.get_source_cluster().get_rank() < 0 || B.get_target_cluster().get_rank() < 0 || B.get_source_cluster().get_rank() < 0);

        std::vector<LowRankMatrix<CoefficientPrecision, CoordinatePrecision>>
            low_rank_matrices;
        std::vector<const Cluster<CoordinatePrecision> *> output_clusters, middle_clusters, input_clusters;
        auto get_output_cluster_A{&HMatrix<CoefficientPrecision, CoordinatePrecision>::get_target_cluster};
        auto get_input_cluster_A{&HMatrix<CoefficientPrecision, CoordinatePrecision>::get_source_cluster};

        const Cluster<CoordinatePrecision> &output_cluster = (A.*get_output_cluster_A)();
        const Cluster<CoordinatePrecision> &middle_cluster = (A.*get_input_cluster_A)();
        const Cluster<CoordinatePrecision> &input_cluster  = B.get_source_cluster();

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
                    const HMatrix<CoefficientPrecision, CoordinatePrecision> *A_child, *B_child;
                    HMatrix<CoefficientPrecision, CoordinatePrecision> *C_child;
                    char actual_transa;
                    if (output_cluster_child == middle_cluster_child) {
                        A_child = (transa == 'N') ? A.get_child_or_this(*output_cluster_child, *middle_cluster_child) : A.get_child_or_this(*middle_cluster_child, *output_cluster_child);
                        B_child = B.get_child_or_this(*middle_cluster_child, *input_cluster_child);
                        C_child = C.get_child_or_this(*output_cluster_child, *input_cluster_child);
                        add_hmatrix_hmatrix_product_symmetry(side, transa, transb, alpha, *A_child, *B_child, CoefficientPrecision(1), *C_child, UPLO, symm);
                    } else if (output_cluster_child->get_offset() >= middle_cluster_child->get_offset() + middle_cluster_child->get_size()) {
                        A_child       = (A.get_UPLO() == 'L') ? A.get_child_or_this(*output_cluster_child, *middle_cluster_child) : A.get_child_or_this(*middle_cluster_child, *output_cluster_child);
                        B_child       = B.get_child_or_this(*middle_cluster_child, *input_cluster_child);
                        C_child       = C.get_child_or_this(*output_cluster_child, *input_cluster_child);
                        actual_transa = (A.get_UPLO() == 'L') ? 'N' : 'T';
                        add_hmatrix_hmatrix_product(actual_transa, transb, alpha, *A_child, *B_child, CoefficientPrecision(1), *C_child);
                    } else {
                        A_child       = (A.get_UPLO() == 'U') ? A.get_sub_hmatrix(*output_cluster_child, *middle_cluster_child) : A.get_sub_hmatrix(*middle_cluster_child, *output_cluster_child);
                        B_child       = B.get_sub_hmatrix(*middle_cluster_child, *input_cluster_child);
                        C_child       = C.get_child_or_this(*output_cluster_child, *input_cluster_child);
                        actual_transa = (A.get_UPLO() == 'U') ? 'N' : 'T';
                        add_hmatrix_hmatrix_product(actual_transa, transb, alpha, *A_child, *B_child, CoefficientPrecision(1), *C_child);
                    }
                }
            }
        }
    } else {
        add_hmatrix_hmatrix_product(transa, transb, alpha, A, B, CoefficientPrecision(1), C);
    }
}

} // namespace htool

#endif
