#ifndef HTOOL_HMATRIX_LINALG_EXECUTOR_FACTORIZATION_HPP
#define HTOOL_HMATRIX_LINALG_EXECUTOR_FACTORIZATION_HPP

#include "../../../basic_types/tree.hpp"               // for preorder_tree_traversal
#include "../../../clustering/cluster_node.hpp"        // for Cluster
#include "../../../matrix/linalg/factorization.hpp"    // for lu_factorization
#include "../../../matrix/matrix.hpp"                  // for Matrix
#include "../../../misc/logger.hpp"                    // for Logger
#include "../../hmatrix.hpp"                           // for HMatrix
#include "../../lrmat/lrmat.hpp"                       // for LowRankMatrix
#include "../omp_task/add_hmatrix_hmatrix_product.hpp" // for fill_clusters
#include "../seq/add_hmatrix_hmatrix_product.hpp"      // for internal_add_hmatrix_hmatrix_product
#include "../seq/add_lrmat_hmatrix.hpp"                // for internal_add_lrmat_hmatrix
#include "../seq/triangular_hmatrix_hmatrix_solve.hpp" // for internal_triangular_hmatrix_hmatrix_solve
#include "thread_executor.hpp"                         // for ExecutorAccess
#include <memory>                                      // for shared_ptr
#include <vector>                                      // for vector

namespace htool {

/**
 * @brief The classical (eager) hierarchical LU factorization of sequential_lu_factorization, with its numeric work run by an executor.
 *
 * The driver follows the recursion of sequential_lu_factorization, internal_triangular_hmatrix_hmatrix_solve and
 * internal_add_hmatrix_hmatrix_product down to the operations on a leaf, and hands each of them to the executor as a task, which calls the
 * same sequential kernels: the dense LU of a diagonal leaf, a triangular solve on a leaf, a product added to a leaf, and, for a product with
 * a leaf factor added to an internal node, its low-rank form, then its restriction added to each leaf. Every leaf is thus updated by the same
 * sequence of operations as in sequential_lu_factorization, so that the factors are the same.
 */
template <typename Executor, typename CoefficientPrecision, typename CoordinatePrecision>
class EagerLUDriver {
    using HMatrixType = HMatrix<CoefficientPrecision, CoordinatePrecision>;
    using ClusterType = Cluster<CoordinatePrecision>;
    using Access      = ExecutorAccess<CoefficientPrecision, CoordinatePrecision>;

    Executor &m_executor;

    static std::vector<const ClusterType *> children_of(const ClusterType &cluster) {
        std::vector<const ClusterType *> clusters;
        fill_clusters(cluster, false, clusters);
        return clusters;
    }

  public:
    explicit EagerLUDriver(Executor &executor) : m_executor(executor) {}

    void lu(HMatrixType &hmatrix) {
        if (hmatrix.is_hierarchical()) {
            std::vector<const ClusterType *> clusters = children_of(hmatrix.get_target_cluster());
            for (std::size_t p = 0; p < clusters.size(); p++) {
                HMatrixType *pivot = hmatrix.get_child_or_this(*clusters[p], *clusters[p]);
                lu(*pivot);
                for (std::size_t q = p + 1; q < clusters.size(); q++) {
                    solve('L', *pivot, *hmatrix.get_child_or_this(*clusters[p], *clusters[q]));
                    solve('R', *pivot, *hmatrix.get_child_or_this(*clusters[q], *clusters[p]));
                }
                for (std::size_t q = p + 1; q < clusters.size(); q++) {
                    for (std::size_t r = p + 1; r < clusters.size(); r++) {
                        add_product(*hmatrix.get_child_or_this(*clusters[q], *clusters[p]), *hmatrix.get_child_or_this(*clusters[p], *clusters[r]), *hmatrix.get_child_or_this(*clusters[q], *clusters[r]));
                    }
                }
            }
        } else if (hmatrix.is_dense()) {
            HMatrixType *leaf = &hmatrix;
            m_executor.spawn([leaf]() { lu_factorization(*leaf->get_dense_data()); }, {}, {Access(*leaf)});
        } else {
            Logger::get_instance().log(LogLevel::ERROR, "Operation is not implemented for lu_factorization (hmatrix is low-rank)"); // LCOV_EXCL_LINE
        }
    }

    // B <- A^-1 B (side='L', with the unit lower triangle of A) or B <- B A^-1 (side='R', with the upper triangle of A), as
    // internal_triangular_hmatrix_hmatrix_solve, whose loops these are for these two cases.
    void solve(char side, const HMatrixType &A, HMatrixType &B) {
        if (A.is_hierarchical() && B.is_hierarchical()) {
            std::vector<const ClusterType *> output_clusters = children_of(side == 'L' ? A.get_target_cluster() : B.get_target_cluster());
            std::vector<const ClusterType *> middle_clusters = children_of(A.get_target_cluster());
            std::vector<const ClusterType *> input_clusters  = children_of(side == 'L' ? B.get_source_cluster() : A.get_source_cluster());
            if (side == 'L') {
                for (const ClusterType *output_cluster : output_clusters) {
                    for (const ClusterType *input_cluster : input_clusters) {
                        HMatrixType *B_child = B.get_child_or_this(*output_cluster, *input_cluster);
                        for (const ClusterType *middle_cluster : middle_clusters) {
                            const HMatrixType *A_child = A.get_child_or_this(*output_cluster, *middle_cluster);
                            if (*output_cluster == *middle_cluster) {
                                solve(side, *A_child, *B_child);
                            } else if (output_cluster->get_offset() > middle_cluster->get_offset()) {
                                add_product(*A_child, *B.get_child_or_this(*middle_cluster, *input_cluster), *B_child);
                            }
                        }
                    }
                }
            } else {
                for (const ClusterType *input_cluster : input_clusters) {
                    for (const ClusterType *output_cluster : output_clusters) {
                        HMatrixType *B_child = B.get_child_or_this(*output_cluster, *input_cluster);
                        for (const ClusterType *middle_cluster : middle_clusters) {
                            const HMatrixType *A_child = A.get_child_or_this(*middle_cluster, *input_cluster);
                            if (*input_cluster == *middle_cluster) {
                                solve(side, *A_child, *B_child);
                            } else if (middle_cluster->get_offset() < input_cluster->get_offset()) {
                                add_product(*B.get_child_or_this(*output_cluster, *middle_cluster), *A_child, *B_child);
                            }
                        }
                    }
                }
            }
        } else if (B.is_hierarchical()) {
            // A is a dense diagonal block, whose cluster is kept by the children of B in a consistent block tree
            for (auto &child : B.get_children_with_ownership()) {
                solve(side, A, *child);
            }
        } else {
            const HMatrixType *A_ptr = &A;
            HMatrixType *B_ptr       = &B;
            char UPLO                = side == 'L' ? 'L' : 'U';
            char diag                = side == 'L' ? 'U' : 'N';
            m_executor.spawn([=]() { internal_triangular_hmatrix_hmatrix_solve(side, UPLO, 'N', diag, CoefficientPrecision(1), *A_ptr, *B_ptr); }, {Access(A)}, {Access(B)});
        }
    }

    // C <- C - A*B, as internal_add_hmatrix_hmatrix_product
    void add_product(const HMatrixType &A, const HMatrixType &B, HMatrixType &C) {
        const HMatrixType *A_ptr = &A;
        const HMatrixType *B_ptr = &B;
        HMatrixType *C_ptr       = &C;
        if (!C.is_hierarchical()) {
            m_executor.spawn([=]() { internal_add_hmatrix_hmatrix_product('N', 'N', CoefficientPrecision(-1), *A_ptr, *B_ptr, CoefficientPrecision(1), *C_ptr); }, {Access(A), Access(B)}, {Access(C)});
        } else if (A.is_hierarchical() && B.is_hierarchical()) {
            std::vector<const ClusterType *> output_clusters = children_of(A.get_target_cluster());
            std::vector<const ClusterType *> middle_clusters = children_of(A.get_source_cluster());
            std::vector<const ClusterType *> input_clusters  = children_of(B.get_source_cluster());
            for (const ClusterType *output_cluster : output_clusters) {
                for (const ClusterType *input_cluster : input_clusters) {
                    for (const ClusterType *middle_cluster : middle_clusters) {
                        const HMatrixType *A_child = A.get_child_or_this(*output_cluster, *middle_cluster);
                        const HMatrixType *B_child = B.get_child_or_this(*middle_cluster, *input_cluster);
                        HMatrixType *C_child       = C.get_child_or_this(*output_cluster, *input_cluster);
                        if (A_child != nullptr && B_child != nullptr && C_child != nullptr) {
                            add_product(*A_child, *B_child, *C_child);
                        }
                    }
                }
            }
        } else {
            // A or B is a leaf: the product has a low-rank form, computed once, whose restrictions are added to the leaves of C
            auto epsilon = A.is_low_rank() ? A.get_low_rank_data()->get_epsilon() : (B.is_low_rank() ? B.get_low_rank_data()->get_epsilon() : C.get_epsilon());
            auto AB      = std::make_shared<LowRankMatrix<CoefficientPrecision>>(A.nb_rows(), B.nb_cols(), epsilon);
            m_executor.spawn([=]() { internal_add_hmatrix_hmatrix_product('N', 'N', CoefficientPrecision(-1), *A_ptr, *B_ptr, CoefficientPrecision(1), *AB); }, {Access(A), Access(B)}, {Access(AB.get())});
            const ClusterType *output_cluster = &C.get_target_cluster();
            const ClusterType *input_cluster  = &C.get_source_cluster();
            preorder_tree_traversal(C, [&](HMatrixType &node) {
                if (!node.is_hierarchical()) {
                    HMatrixType *leaf = &node;
                    m_executor.spawn([=]() { internal_add_lrmat_hmatrix(*AB, *output_cluster, *input_cluster, *leaf); }, {Access(AB.get())}, {Access(node)});
                }
            });
        }
    }
};

/**
 * @brief LU factorization of hmatrix, as sequential_lu_factorization, with its numeric work run by executor (see EagerLUDriver). Returns
 * once the factorization is done.
 */
template <typename Executor, typename CoefficientPrecision, typename CoordinatePrecision>
void executor_lu_factorization(Executor &executor, HMatrix<CoefficientPrecision, CoordinatePrecision> &hmatrix) {
    if (!hmatrix.is_block_tree_consistent()) {
        Logger::get_instance().log(LogLevel::ERROR, "executor_lu_factorization is only implemented for consistent block tree."); // LCOV_EXCL_LINE
    }
    EagerLUDriver<Executor, CoefficientPrecision, CoordinatePrecision> driver(executor);
    driver.lu(hmatrix);
    executor.wait();
}

} // namespace htool

#endif
