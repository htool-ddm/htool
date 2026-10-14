#ifndef HTOOL_HMATRIX_LINALG_SEQ_SOLVE_HPP
#define HTOOL_HMATRIX_LINALG_SEQ_SOLVE_HPP

#include "../../../matrix/matrix.hpp"
#include "../../../misc/misc.hpp"
#include "../../hmatrix.hpp"
#include "apply_ldlt_diagonal.hpp"
#include "triangular_hmatrix_matrix_solve.hpp"
#include "triangular_ldlt_hmatrix_matrix_solve.hpp"

namespace htool {

template <typename Mat, typename CoordinatePrecision = underlying_type<typename Mat::value_type>>
void internal_lu_solve(char trans, const HMatrix<typename Mat::value_type, CoordinatePrecision> &A, Mat &X) {
    using CoefficientPrecision = typename Mat::value_type;
    if (trans == 'N') {
        internal_triangular_hmatrix_matrix_solve('L', 'L', 'N', 'U', CoefficientPrecision(1), A, X);
        internal_triangular_hmatrix_matrix_solve('L', 'U', 'N', 'N', CoefficientPrecision(1), A, X);
    } else {
        internal_triangular_hmatrix_matrix_solve('L', 'U', trans, 'N', CoefficientPrecision(1), A, X);
        internal_triangular_hmatrix_matrix_solve('L', 'L', trans, 'U', CoefficientPrecision(1), A, X);
    }
}

template <typename CoefficientPrecision, typename CoordinatePrecision = underlying_type<CoefficientPrecision>>
void internal_cholesky_solve(char UPLO, const HMatrix<CoefficientPrecision, CoordinatePrecision> &A, Matrix<CoefficientPrecision> &X) {
    if (UPLO == 'L') {
        internal_triangular_hmatrix_matrix_solve('L', 'L', 'N', 'N', CoefficientPrecision(1), A, X);
        internal_triangular_hmatrix_matrix_solve('L', 'L', is_complex<CoefficientPrecision>() ? 'C' : 'T', 'N', CoefficientPrecision(1), A, X);
    } else {
        internal_triangular_hmatrix_matrix_solve('L', 'U', is_complex<CoefficientPrecision>() ? 'C' : 'T', 'N', CoefficientPrecision(1), A, X);
        internal_triangular_hmatrix_matrix_solve('L', 'U', 'N', 'N', CoefficientPrecision(1), A, X);
    }
}

// A = L D L^T (L^H for symmetry='H'), so X := L^-T D^-1 L^-1 X, as sytrs/hetrs.
template <typename CoefficientPrecision, typename CoordinatePrecision>
void internal_ldlt_solve(char symmetry, char UPLO, const HMatrix<CoefficientPrecision, CoordinatePrecision> &A, Matrix<CoefficientPrecision> &X) {
    internal_triangular_ldlt_hmatrix_matrix_solve('L', UPLO, 'N', CoefficientPrecision(1), A, X);
    internal_apply_ldlt_diagonal(symmetry, 'L', UPLO, A, X, A.get_target_cluster().get_offset());
    internal_triangular_ldlt_hmatrix_matrix_solve('L', UPLO, symmetry == 'H' ? 'C' : 'T', CoefficientPrecision(1), A, X);
}

} // namespace htool

#endif
