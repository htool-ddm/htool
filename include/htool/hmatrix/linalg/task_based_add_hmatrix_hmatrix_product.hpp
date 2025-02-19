#ifndef HTOOL_TASK_BASED_HMATRIX_LINALG_ADD_HMATRIX_HMATRIX_PRODUCT_HPP
#define HTOOL_TASK_BASED_HMATRIX_LINALG_ADD_HMATRIX_HMATRIX_PRODUCT_HPP

// #include "../../matrix/linalg/add_matrix_vector_product.hpp"      // for add_mat...
// #include "../../misc/misc.hpp"                                    // for underly...
#include "../../wrappers/wrapper_blas.hpp" // for Blas
#include "../hmatrix.hpp"                  // for HMatrix
#include "add_lrmat_hmatrix.hpp"           // for internal_add_lrmat_hmatrix
// #include "../lrmat/linalg/add_lrmat_vector_product.hpp"           // for add_lrm...
#include <algorithm> // for transform, max
// #include <complex>                                                // for complex
#include <htool/hmatrix/tree_builder/task_based_tree_builder.hpp> // for enumerate_dependence, find_l0...
#include <mutex>
#include <unistd.h>
#include <vector> // for vector

namespace htool {

/**
 * \brief Task-based internal addition of HMatrix-HMatrix product.
 *
 * This function computes the product of two HMatrix objects, `A` and `B`
 * scaled by `alpha`, and adds the result to a third HMatrix object, `C`,
 * scaled by `beta`. The computation is performed in a task-based manner using
 * OpenMP tasks to parallelize the operations over the hierarchical structure
 * of the matrices.
 *
 * \param transa A character indicating the transpose operation on matrix `A`.
 * \param transb A character indicating the transpose operation on matrix `B`.
 * \param alpha A scalar coefficient to scale the product of `A` and `B`.
 * \param A The first input HMatrix.
 * \param B The second input HMatrix.
 * \param beta A scalar coefficient to scale the matrix `C`.
 * \param C The output HMatrix to which the result is added.
 * \param L0 A vector of pointers to nodes of the output HMatrix `C`.
 * \param L0_A A vector of pointers to nodes of the input HMatrix `A`.
 * \param L0_B A vector of pointers to nodes of the input HMatrix `B`.
 *
 * The function handles different types of matrix structures (dense, low-rank,
 * hierarchical) and performs recursive calls if necessary to handle non-leaf
 * nodes. It uses dependency tracking to ensure correct task execution order.
 */

template <typename CoefficientPrecision, typename CoordinatePrecision = CoefficientPrecision>
void task_based_internal_add_hmatrix_hmatrix_product(char transa, char transb, CoefficientPrecision alpha, const HMatrix<CoefficientPrecision, CoordinatePrecision> &A, const HMatrix<CoefficientPrecision, CoordinatePrecision> &B, CoefficientPrecision beta, HMatrix<CoefficientPrecision, CoordinatePrecision> &C, std::vector<HMatrix<CoefficientPrecision> *> &L0_A, std::vector<HMatrix<CoefficientPrecision> *> &L0_B, std::vector<HMatrix<CoefficientPrecision> *> &L0) {

    // int max_prio = std::max(0, omp_get_max_task_priority());

    // Scale the output vector `out` with `beta` if `beta` is not equal to 1
    if (CoefficientPrecision(beta) != CoefficientPrecision(1)) {
        for (auto &L0C_node : L0) {

#if defined(_OPENMP) && !defined(HTOOL_WITH_PYTHON_INTERFACE)
#    pragma omp task default(none)   \
        firstprivate(L0C_node, beta) \
        depend(inout : *L0C_node)
#endif
            {
                sequential_scale(beta, *L0C_node);
            }
        }
    }

    // check if C is in L0.
    bool is_C_in_L0 = false;
    for (auto &L0C_node : L0) {
        if (L0C_node->get_target_cluster() == C.get_target_cluster() && L0C_node->get_source_cluster() == C.get_source_cluster()) {
            is_C_in_L0 = true;
            break;
        }
    }

    // Fill the clusters for the output, middle, and input clusters
    bool block_tree_not_consistent = (A.get_target_cluster().get_rank() < 0 || A.get_source_cluster().get_rank() < 0 || B.get_target_cluster().get_rank() < 0 || B.get_source_cluster().get_rank() < 0);

    std::vector<const Cluster<CoordinatePrecision> *> output_clusters, middle_clusters, input_clusters;
    const Cluster<CoordinatePrecision> &output_cluster = transa == 'N' ? A.get_target_cluster() : A.get_source_cluster();
    const Cluster<CoordinatePrecision> &middle_cluster = transa == 'N' ? A.get_source_cluster() : A.get_target_cluster();
    const Cluster<CoordinatePrecision> &input_cluster  = transb == 'N' ? B.get_source_cluster() : B.get_target_cluster();

    fill_clusters(output_cluster, block_tree_not_consistent, output_clusters);
    fill_clusters(input_cluster, block_tree_not_consistent, input_clusters);
    fill_clusters(middle_cluster, block_tree_not_consistent, middle_clusters);

    // C += alpha A * B
    if (!is_C_in_L0 && !A.is_leaf() && !B.is_leaf() && !C.is_leaf()) { // recursive call
        // std::cout << "Recursive call" << std::endl;
        for (auto &output_cluster_child : output_clusters) {
            for (auto &input_cluster_child : input_clusters) {
                for (auto &middle_cluster_child : middle_clusters) {
                    auto &&A_child = (transa == 'N') ? A.get_child_or_this(*output_cluster_child, *middle_cluster_child) : A.get_child_or_this(*middle_cluster_child, *output_cluster_child);
                    auto &&B_child = (transb == 'N') ? B.get_child_or_this(*middle_cluster_child, *input_cluster_child) : B.get_child_or_this(*input_cluster_child, *middle_cluster_child);
                    auto &&C_child = C.get_child_or_this(*output_cluster_child, *input_cluster_child);
                    if (A_child != nullptr && B_child != nullptr && C_child != nullptr) {
                        task_based_internal_add_hmatrix_hmatrix_product(transa, transb, alpha, *A_child, *B_child, CoefficientPrecision(1), *C.get_child_or_this(*output_cluster_child, *input_cluster_child), L0_A, L0_B, L0);
                    }
                }
            }
        }
    } else { // if (is_C_in_L0 || A.is_leaf() || B.is_leaf() ||C.is_leaf())
        std::vector<const HMatrix<CoefficientPrecision, CoordinatePrecision> *> read_deps, temp;
        read_deps = enumerate_dependences(A, L0_A);
        temp      = enumerate_dependences(B, L0_B);
        read_deps.insert(read_deps.end(), temp.begin(), temp.end()); // concatenating the two vectors
        auto read_deps_size = read_deps.size();

        if (is_C_in_L0) {
#if defined(_OPENMP) && !defined(HTOOL_WITH_PYTHON_INTERFACE)
#    pragma omp task default(none)                                     \
        firstprivate(transa, transb, alpha, read_deps_size)            \
        shared(A, B, C, read_deps)                                     \
        depend(iterator(it = 0 : read_deps_size), in : *read_deps[it]) \
        depend(inout : C)
#endif
            {
                internal_add_hmatrix_hmatrix_product(transa, transb, alpha, A, B, CoefficientPrecision(1), C);
            }

        } else {
            auto AB = std::make_shared<LowRankMatrix<CoefficientPrecision>>(A.nb_rows(), B.nb_cols(), C.get_epsilon());

#if defined(_OPENMP) && !defined(HTOOL_WITH_PYTHON_INTERFACE)
#    pragma omp task default(none)                              \
        firstprivate(transa, transb, alpha, AB, read_deps_size) \
        shared(A, B, read_deps)                                 \
        depend(inout : *AB)                                     \
        depend(iterator(it = 0 : read_deps_size), in : *read_deps[it])
#endif
            {
                if (A.is_low_rank() || B.is_low_rank()) {
                    if (A.is_dense() and B.is_low_rank()) {
                        add_matrix_lrmat_product(transa, transb, alpha, *A.get_dense_data(), *B.get_low_rank_data(), CoefficientPrecision(1), *AB);
                    } else if (A.is_hierarchical() and B.is_low_rank()) {
                        internal_add_hmatrix_lrmat_product(transa, transb, alpha, A, *B.get_low_rank_data(), CoefficientPrecision(1), *AB);
                    } else if (A.is_low_rank() and B.is_low_rank()) {
                        add_lrmat_lrmat_product(transa, transb, alpha, *A.get_low_rank_data(), *B.get_low_rank_data(), CoefficientPrecision(1), *AB);
                    } else if (A.is_low_rank() and B.is_dense()) {
                        add_lrmat_matrix_product(transa, transb, alpha, *A.get_low_rank_data(), *B.get_dense_data(), CoefficientPrecision(1), *AB);
                    } else if (A.is_low_rank() and B.is_hierarchical()) {
                        internal_add_lrmat_hmatrix_product(transa, transb, alpha, *A.get_low_rank_data(), B, CoefficientPrecision(1), *AB);
                    }
                } else {
                    if (A.is_dense() and B.is_dense()) {
                        add_matrix_matrix_product(transa, transb, alpha, *A.get_dense_data(), *B.get_dense_data(), CoefficientPrecision(1), *AB);
                    } else if (A.is_dense() and B.is_hierarchical()) {
                        internal_add_matrix_hmatrix_product(transa, transb, alpha, *A.get_dense_data(), B, CoefficientPrecision(1), *AB);
                    } else if (A.is_hierarchical() and B.is_dense()) {
                        internal_add_hmatrix_matrix_product(transa, transb, alpha, A, *B.get_dense_data(), CoefficientPrecision(1), *AB);
                    }
                }
            }
            // Use a ptr to keep output_cluster and input_cluster alive as long as any task is running
            const Cluster<CoordinatePrecision> *output_cluster_ptr = &output_cluster;
            const Cluster<CoordinatePrecision> *input_cluster_ptr  = &input_cluster;
            auto write_deps                                        = enumerate_dependences(C, L0);

            for (auto &C_node : write_deps) {
#if defined(_OPENMP) && !defined(HTOOL_WITH_PYTHON_INTERFACE)
#    pragma omp task default(none)                                                  \
        firstprivate(AB, C_node, output_cluster_ptr, input_cluster_ptr, write_deps) \
        depend(inout : *C_node)                                                     \
        depend(in : *AB)
#endif
                {
                    internal_add_lrmat_hmatrix(*AB, *output_cluster_ptr, *input_cluster_ptr, *C_node);
                }
            }
        } // end of else (is_C_in_L0)
    } // end of if (!is_C_in_L0 && A.is_leaf() && B.is_leaf() && C.is_leaf())
} // end of task_based_internal_add_hmatrix_hmatrix_product

template <typename CoordinatePrecision>
void fill_clusters(const Cluster<CoordinatePrecision> &cluster, bool block_tree_not_consistent, std::vector<const Cluster<CoordinatePrecision> *> &clusters) {
    if (cluster.is_leaf() || (block_tree_not_consistent && cluster.get_rank() >= 0)) {
        clusters.push_back(&cluster);
    } else if (block_tree_not_consistent) {
        for (auto &child : cluster.get_clusters_on_partition()) {
            clusters.push_back(child);
        }
    } else {
        for (auto &child : cluster.get_children()) {
            clusters.push_back(child.get());
        }
    }
}

} // namespace htool

#endif
