#ifndef HTOOL_HMATRIX_LINALG_TRIANGULAR_LDLT_HMATRIX_LRMAT_SOLVE_HPP
#define HTOOL_HMATRIX_LINALG_TRIANGULAR_LDLT_HMATRIX_LRMAT_SOLVE_HPP

#include "../../misc/misc.hpp"
#include "../hmatrix.hpp"
#include "../lrmat/lrmat.hpp"
#include "triangular_ldlt_hmatrix_matrix_solve.hpp"

namespace htool {
template <typename CoefficientPrecision, typename CoordinatePrecision = underlying_type<CoefficientPrecision>>
void internal_triangular_ldlt_hmatrix_lrmat_solve(char side, char UPLO, char transa, CoefficientPrecision alpha, const HMatrix<CoefficientPrecision, CoordinatePrecision> &A, LowRankMatrix<CoefficientPrecision> &B) {
    if (alpha != CoefficientPrecision(1)) {
        scale(alpha, B);
    }
    if (side == 'L' or side == 'l') {
        internal_triangular_ldlt_hmatrix_matrix_solve('L', UPLO, transa, CoefficientPrecision(1), A, B.get_U());
    } else {
        internal_triangular_ldlt_hmatrix_matrix_solve('R', UPLO, transa, CoefficientPrecision(1), A, B.get_V());
    }
}

} // namespace htool
#endif
