#ifndef HTOOL_HMATRIX_UTILS_RECOMPRESSION_HPP
#define HTOOL_HMATRIX_UTILS_RECOMPRESSION_HPP
#include "../hmatrix.hpp"
#include "../lrmat/utils/SVD_recompression.hpp"
namespace htool {

template <typename CoefficientPrecision, typename CoordinatePrecision = underlying_type<CoefficientPrecision>, class Recompression = decltype(SVD_recompression<CoefficientPrecision>(std::declval<LowRankMatrix<CoefficientPrecision> &>()))(LowRankMatrix<CoefficientPrecision> &)>
void recompression(HMatrix<CoefficientPrecision, CoordinatePrecision> &hmatrix, Recompression recompression = &SVD_recompression<CoefficientPrecision>) {
    auto leaves = get_low_rank_leaves_from(hmatrix); // C++17 structured binding
    for (auto &leaf : leaves) {
        recompression(*leaf->get_low_rank_data());
    }
}

template <typename CoefficientPrecision, typename CoordinatePrecision = underlying_type<CoefficientPrecision>, class Recompression = decltype(SVD_recompression<CoefficientPrecision>(std::declval<LowRankMatrix<CoefficientPrecision> &>()))(LowRankMatrix<CoefficientPrecision> &)>
void openmp_recompression(HMatrix<CoefficientPrecision, CoordinatePrecision> &hmatrix, Recompression recompression = &SVD_recompression<CoefficientPrecision>) {
    auto leaves = get_low_rank_leaves_from(hmatrix); // C++17 structured binding

#if defined(_OPENMP)
#    pragma omp parallel
#endif
    {
#if defined(_OPENMP)
#    pragma omp for schedule(guided) nowait
#endif
        for (int i = 0; i < leaves.size(); i++) {
            recompression(*leaves[i]->get_low_rank_data());
        }
    }
}
} // namespace htool

#endif
