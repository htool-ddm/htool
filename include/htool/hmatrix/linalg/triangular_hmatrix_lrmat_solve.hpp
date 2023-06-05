#ifndef HTOOL_HMATRIX_LINALG_TRIANGULAR_HMATRIX_LRMAT_SOLVE_HPP
#define HTOOL_HMATRIX_LINALG_TRIANGULAR_HMATRIX_LRMAT_SOLVE_HPP

#include "../../misc/misc.hpp"                 // for underlying_type
#include "../hmatrix.hpp"                      // for HMatrix
#include "../lrmat/lrmat.hpp"                  // for LowRankMatrix
#include "triangular_hmatrix_matrix_solve.hpp" // for triangular_hmatrix_...

namespace htool {
template <typename CoefficientPrecision, typename CoordinatePrecision = underlying_type<CoefficientPrecision>>
void internal_triangular_hmatrix_lrmat_solve(char side, char UPLO, char transa, char diag, CoefficientPrecision alpha, const HMatrix<CoefficientPrecision, CoordinatePrecision> &A, LowRankMatrix<CoefficientPrecision, CoordinatePrecision> &B) {
    if (alpha != CoefficientPrecision(1)) {
        scale(alpha, B);
    }
    if (side == 'L' or side == 'l') {
        internal_triangular_hmatrix_matrix_solve('L', UPLO, transa, diag, CoefficientPrecision(1), A, B.get_U());
    } else {
        internal_triangular_hmatrix_matrix_solve('R', UPLO, transa, diag, CoefficientPrecision(1), A, B.get_V());
    }
}

} // namespace htool
#endif
