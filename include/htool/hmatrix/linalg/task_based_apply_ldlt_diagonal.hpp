#ifndef HTOOL_TASK_BASED_HMATRIX_LINALG_APPLY_LDLT_DIAGONAL_HPP
#define HTOOL_TASK_BASED_HMATRIX_LINALG_APPLY_LDLT_DIAGONAL_HPP

#include "../../misc/logger.hpp"
#include "../hmatrix.hpp"
#include "../task_dependencies.hpp"
#include "apply_ldlt_diagonal.hpp"
#include <vector>

// to remove warning from depend(iterator(it = 0 : read_deps_size), in : *read_deps[it])
#if defined(__clang__)
#elif defined(__GNUC__) || defined(__GNUG__)
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wuseless-cast"
#endif

namespace htool {

/**
 * @brief Task-based internal_apply_ldlt_diagonal: B := D^-1 B (side='L') or B D^-1 (side='R'), with one task per node of L0_B in B.
 *
 * @param[in] A The HMatrix holding D at its dense diagonal leaves.
 * @param[in,out] B The HMatrix \f$B\f$.
 * @param[in] L0_A The L0 of \f$A\f$.
 * @param[in] L0_B The L0 of \f$B\f$.
 */
template <typename CoefficientPrecision, typename CoordinatePrecision>
void task_based_internal_apply_ldlt_diagonal(char symmetry, char side, char UPLO, const HMatrix<CoefficientPrecision, CoordinatePrecision> &A, HMatrix<CoefficientPrecision, CoordinatePrecision> &B, std::vector<HMatrix<CoefficientPrecision> *> &L0_A, std::vector<HMatrix<CoefficientPrecision> *> &L0_B) {
    for (HMatrix<CoefficientPrecision, CoordinatePrecision> *L0_node : enumerate_dependences(B, L0_B)) {
        // B below L0_B: L0_node is its ancestor, so the task only modifies B
        HMatrix<CoefficientPrecision, CoordinatePrecision> *B_node = left_hmatrix_ancestor_of_right_hmatrix(B, *L0_node) ? L0_node : &B;

        const Cluster<CoordinatePrecision> &cluster                      = side == 'L' ? B_node->get_target_cluster() : B_node->get_source_cluster();
        const HMatrix<CoefficientPrecision, CoordinatePrecision> *A_node = A.get_sub_hmatrix(cluster, cluster);
        // Cannot happen for a consistent block tree, whose diagonal blocks are refined down to the leaf clusters
        if (A_node == nullptr) {
            htool::Logger::get_instance().log(LogLevel::ERROR, "task_based_internal_apply_ldlt_diagonal: no diagonal block of A matches a block of B in L0."); // LCOV_EXCL_LINE
            continue;                                                                                                                                          // LCOV_EXCL_LINE
        }

#if defined(_OPENMP)
        std::vector<const HMatrix<CoefficientPrecision, CoordinatePrecision> *> read_deps = enumerate_dependences(*A_node, L0_A);
        int read_deps_size                                                                = read_deps.size();
#    pragma omp task default(none)                         \
        firstprivate(symmetry, side, UPLO, A_node, B_node) \
        shared(read_deps)                                  \
        depend(inout : *L0_node)                           \
        depend(iterator(it = 0 : read_deps_size), in : *read_deps[it])
#endif
        {
            internal_apply_ldlt_diagonal(symmetry, side, UPLO, *A_node, *B_node);
        }
    }
}

} // namespace htool

#if defined(__clang__)
#elif defined(__GNUC__) || defined(__GNUG__)
#    pragma GCC diagnostic pop
#endif
#endif
