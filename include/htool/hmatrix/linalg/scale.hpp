#ifndef HTOOL_HMATRIX_LINALG_SCALE_HPP
#define HTOOL_HMATRIX_LINALG_SCALE_HPP

#include "../hmatrix.hpp" // for HMatrix
#include "../lrmat/linalg/scale.hpp"
#include "htool/basic_types/tree.hpp" // for preorder_tree_traversal
#include <vector>                     // for vector

namespace htool {

template <typename CoefficientPrecision, typename CoordinatePrecision = underlying_type<CoefficientPrecision>>
void openmp_scale(CoefficientPrecision da, HMatrix<CoefficientPrecision, CoordinatePrecision> &A) {
    std::vector<HMatrix<CoefficientPrecision, CoordinatePrecision> *> leaves;
    preorder_tree_traversal(A, [&leaves](HMatrix<CoefficientPrecision, CoordinatePrecision> &current_node) {
        if (!current_node.is_hierarchical()) {
            leaves.push_back(&current_node);
        }
    });

#if defined(_OPENMP)
#    pragma omp parallel
#endif
    {
#if defined(_OPENMP)
#    pragma omp for schedule(guided) nowait
#endif
        for (int b = 0; b < leaves.size(); b++) {
            if (leaves[b]->is_dense()) {
                scale(da, *leaves[b]->get_dense_data());
            } else {
                scale(da, *leaves[b]->get_low_rank_data());
            }
        }
    }
}

template <typename CoefficientPrecision, typename CoordinatePrecision = underlying_type<CoefficientPrecision>>
void sequential_scale(CoefficientPrecision da, HMatrix<CoefficientPrecision, CoordinatePrecision> &A) {
    std::vector<HMatrix<CoefficientPrecision, CoordinatePrecision> *> leaves;
    preorder_tree_traversal(A, [&leaves](HMatrix<CoefficientPrecision, CoordinatePrecision> &current_node) {
        if (!current_node.is_hierarchical()) {
            leaves.push_back(&current_node);
        }
    });
    for (int b = 0; b < leaves.size(); b++) {
        if (leaves[b]->is_dense()) {
            scale(da, *leaves[b]->get_dense_data());
        } else {
            scale(da, *leaves[b]->get_low_rank_data());
        }
    }
}

template <typename CoefficientPrecision, typename CoordinatePrecision = underlying_type<CoefficientPrecision>>
void scale(CoefficientPrecision da, HMatrix<CoefficientPrecision, CoordinatePrecision> &A) {
    openmp_scale(da, A);
}

} // namespace htool

#endif
