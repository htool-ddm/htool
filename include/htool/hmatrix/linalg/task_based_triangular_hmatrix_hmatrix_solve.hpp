#ifndef HTOOL_TASK_BASED_HMATRIX_LINALG_TRIANGULAR_HMATRIX_HMATRIX_SOLVE_HPP
#define HTOOL_TASK_BASED_HMATRIX_LINALG_TRIANGULAR_HMATRIX_HMATRIX_SOLVE_HPP

#include "../../clustering/cluster_node.hpp"
#include "../../matrix/linalg/factorization.hpp"
#include "../../matrix/matrix.hpp"
#include "../../misc/logger.hpp"
#include "../../misc/misc.hpp"
#include "../hmatrix.hpp"
#include "task_based_add_hmatrix_hmatrix_product.hpp"
#include "triangular_hmatrix_lrmat_solve.hpp"
#include "triangular_hmatrix_matrix_solve.hpp"
#include <memory>
#include <string>
#include <vector>

// to remove warning from depend(iterator(it = 0 : read_deps_size), in : *read_deps[it])
#if defined(__clang__)
#elif defined(__GNUC__) || defined(__GNUG__)
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wuseless-cast"
#endif

namespace htool {
/**
 * @brief Solves a triangular linear system of equation of the form \f$Ax = B\f$ where \f$A\f$ is a triangular HMatrix and \f$B\f$ is an HMatrix. The solve is done in a task-based manner
 * and uses task based HMatrix-HMatrix product.
 *
 * @param[in] side Indicates whether \f$A\f$ is on the left or right side of \f$x\f$.
 * @param[in] UPLO Indicates whether the upper or lower triangular part of \f$A\f$ is used.
 * @param[in] transa Indicates whether the matrix \f$A\f$ is transposed or not.
 * @param[in] diag Is passed to internal_triangular_hmatrix_hmatrix_solve function
 * @param[in] alpha The scalar \f$\alpha\f$.
 * @param[in] A The triangular HMatrix \f$A\f$.
 * @param[in,out] B The HMatrix \f$B\f$.
 * @param[in] L0_A The L0 of \f$A\f$.
 * @param[in] L0_B The L0 of \f$B\f$.
 */
template <typename CoefficientPrecision, typename CoordinatePrecision = underlying_type<CoefficientPrecision>>
void task_based_internal_triangular_hmatrix_hmatrix_solve(char side, char UPLO, char transa, char diag, CoefficientPrecision alpha, const HMatrix<CoefficientPrecision, CoordinatePrecision> &A, HMatrix<CoefficientPrecision, CoordinatePrecision> &B, std::vector<HMatrix<CoefficientPrecision> *> &L0_A, std::vector<HMatrix<CoefficientPrecision> *> &L0_B) {
    // if (alpha != CoefficientPrecision(1)) {
    //     scale(alpha, B);
    // }

    // Scale 'B' with `alpha` if `alpha` is not equal to 1
    if (CoefficientPrecision(alpha) != CoefficientPrecision(1)) {
        for (auto &L0_node : L0_B) {

#if defined(_OPENMP) && !defined(HTOOL_WITH_PYTHON_INTERFACE)
#    pragma omp task default(none)   \
        firstprivate(L0_node, alpha) \
        depend(inout : *L0_node)
#endif
            {
                scale(alpha, *L0_node);
            }
        }
    }

    // check if B is in L0_B.
    bool is_B_in_L0_B = false;
    for (auto &L0_node : L0_B) {
        if (L0_node->get_target_cluster() == B.get_target_cluster() && L0_node->get_source_cluster() == B.get_source_cluster()) {
            is_B_in_L0_B = true;
            break;
        }
    }

    if (is_B_in_L0_B) {
#if defined(_OPENMP) && !defined(HTOOL_WITH_PYTHON_INTERFACE)
        std::vector<const HMatrix<CoefficientPrecision, CoordinatePrecision> *> read_deps = enumerate_dependences(A, L0_A);
        int read_deps_size                                                                = read_deps.size();
#    pragma omp task default(none)                    \
        firstprivate(side, UPLO, transa, alpha, diag) \
        shared(A, B, read_deps)                       \
        depend(in : A)                                \
        depend(inout : B)                             \
        depend(iterator(it = 0 : read_deps_size), in : *read_deps[it])

#endif
        {
            // internal_triangular_hmatrix_hmatrix_solve(side, UPLO, transa, 'N', alpha, A, B);
            internal_triangular_hmatrix_hmatrix_solve(side, UPLO, transa, diag, CoefficientPrecision(1), A, B); // alpha == 1 because the scaling is done here
        }
    } else {
        // Fill the clusters for the output, middle, and input clusters
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

        fill_clusters(output_cluster, block_tree_not_consistent, output_clusters);
        fill_clusters(input_cluster, block_tree_not_consistent, input_clusters);
        fill_clusters(middle_cluster, block_tree_not_consistent, middle_clusters);

        // Forward, compute each block rows one after the other
        if ((UPLO == 'L' && transa == 'N' && side == 'L') || (UPLO == 'U' && transa != 'N' && side == 'L')) {
            // std::cout << "clusters' size: " << output_clusters.size() << ", " << middle_clusters.size() << ", " << input_clusters.size() << std::endl;
            for (auto &output_cluster_child : output_clusters) {
                for (auto &input_cluster_child : input_clusters) {
                    HMatrix<CoefficientPrecision, CoordinatePrecision> *B_child_to_modify = B.get_sub_hmatrix(*output_cluster_child, *input_cluster_child);

                    for (auto &middle_cluster_child : middle_clusters) {
                        const HMatrix<CoefficientPrecision, CoordinatePrecision> *A_child = transa == 'N' ? A.get_sub_hmatrix(*output_cluster_child, *middle_cluster_child) : A.get_sub_hmatrix(*middle_cluster_child, *output_cluster_child);
                        if (*output_cluster_child == *middle_cluster_child) {
                            task_based_internal_triangular_hmatrix_hmatrix_solve(side, UPLO, transa, diag, CoefficientPrecision(1), *A_child, *B_child_to_modify, L0_A, L0_B);
                        } else if (output_cluster_child->get_offset() > middle_cluster_child->get_offset()) {
                            const HMatrix<CoefficientPrecision, CoordinatePrecision> *B_child = B.get_sub_hmatrix(*middle_cluster_child, *input_cluster_child);
                            task_based_internal_add_hmatrix_hmatrix_product(transa, 'N', CoefficientPrecision(-1), *A_child, *B_child, CoefficientPrecision(1), *B_child_to_modify, L0_A, L0_B, L0_B);
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
                            task_based_internal_triangular_hmatrix_hmatrix_solve(side, UPLO, transa, diag, CoefficientPrecision(1), *A_child, *B_child_to_modify, L0_A, L0_B);
                        } else if (output_cluster_child->get_offset() < middle_cluster_child->get_offset()) {

                            const HMatrix<CoefficientPrecision, CoordinatePrecision> *B_child = B.get_sub_hmatrix(*middle_cluster_child, *input_cluster_child);

                            task_based_internal_add_hmatrix_hmatrix_product(transa, 'N', CoefficientPrecision(-1), *A_child, *B_child, CoefficientPrecision(1), *B_child_to_modify, L0_A, L0_B, L0_B);
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
                            task_based_internal_triangular_hmatrix_hmatrix_solve(side, UPLO, transa, diag, CoefficientPrecision(1), *A_child, *B_child_to_modify, L0_A, L0_B);
                        } else if (middle_cluster_child->get_offset() < input_cluster_child->get_offset()) {

                            const HMatrix<CoefficientPrecision, CoordinatePrecision> *B_child = B.get_sub_hmatrix(*output_cluster_child, *middle_cluster_child);

                            task_based_internal_add_hmatrix_hmatrix_product('N', transa, CoefficientPrecision(-1), *B_child, *A_child, CoefficientPrecision(1), *B_child_to_modify, L0_B, L0_A, L0_B);
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
                            task_based_internal_triangular_hmatrix_hmatrix_solve(side, UPLO, transa, diag, CoefficientPrecision(1), *A_child, *B_child_to_modify, L0_A, L0_B);
                        } else if (middle_cluster_child->get_offset() > input_cluster_child->get_offset()) {

                            const HMatrix<CoefficientPrecision, CoordinatePrecision> *B_child = B.get_sub_hmatrix(*output_cluster_child, *middle_cluster_child);

                            task_based_internal_add_hmatrix_hmatrix_product('N', transa, CoefficientPrecision(-1), *B_child, *A_child, CoefficientPrecision(1), *B_child_to_modify, L0_B, L0_A, L0_B);
                        }
                    }
                }
            }
        }
    }
}
} // namespace htool

#if defined(__clang__)
#elif defined(__GNUC__) || defined(__GNUG__)
#    pragma GCC diagnostic pop
#endif
#endif
