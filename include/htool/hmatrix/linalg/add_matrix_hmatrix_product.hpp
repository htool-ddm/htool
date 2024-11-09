#ifndef HTOOL_HMATRIX_LINALG_ADD_MATRIX_HMATRIX_PRODUCT_HPP
#define HTOOL_HMATRIX_LINALG_ADD_MATRIX_HMATRIX_PRODUCT_HPP

#include "../../matrix/linalg/scale.hpp"            // for scale
#include "../../matrix/linalg/transpose.hpp"        // for transpose
#include "../../matrix/matrix.hpp"                  // for Matrix
#include "../../matrix/utils/SVD_truncation.hpp"    // for SVD_truncation
#include "../../misc/misc.hpp"                      // for conj_if_complex
#include "../../wrappers/wrapper_blas.hpp"          // for Blas
#include "../hmatrix.hpp"                           // for HMatrix
#include "../lrmat/lrmat.hpp"                       // for LowRankMatrix
#include "../lrmat/utils/SVD_recompression.hpp"     // for recompression
#include "add_hmatrix_matrix_product_row_major.hpp" // for sequential_ad...
#include <algorithm>                                // for copy_n, min
#include <vector>                                   // for vector

namespace htool {

template <typename CoefficientPrecision, typename CoordinatePrecision = underlying_type<CoefficientPrecision>>
void internal_add_matrix_hmatrix_product(char transa, char transb, CoefficientPrecision alpha, const Matrix<CoefficientPrecision> &A, const HMatrix<CoefficientPrecision, CoordinatePrecision> &B, CoefficientPrecision beta, Matrix<CoefficientPrecision> &C) {

    char new_transa = transb == 'N' ? 'T' : 'N';
    new_transa      = B.get_symmetry_for_leaves() == 'H' and transb == 'N' ? 'N' : new_transa;
    new_transa      = B.get_symmetry_for_leaves() == 'S' and transb == 'N' ? 'N' : new_transa;
    if (transa == 'N') {
        bool need_buffer_for_conj = transb == 'C' or (B.get_symmetry_for_leaves() == 'H' and transb == 'N');
        std::vector<CoefficientPrecision> buffer_A(need_buffer_for_conj ? A.nb_cols() * A.nb_rows() : 0);
        if (need_buffer_for_conj) {
            std::copy(A.data(), A.data() + A.nb_cols() * A.nb_rows(), buffer_A.data());
            conj_if_complex(buffer_A.data(), buffer_A.size());
            conj_if_complex(C.data(), C.nb_rows() * C.nb_cols());
        }
        sequential_internal_add_hmatrix_matrix_product_row_major(new_transa, 'N', need_buffer_for_conj ? conj_if_complex(alpha) : alpha, B, need_buffer_for_conj ? buffer_A.data() : A.data(), need_buffer_for_conj ? conj_if_complex(beta) : beta, C.data(), C.nb_rows());
        if (need_buffer_for_conj) {
            conj_if_complex(C.data(), C.nb_rows() * C.nb_cols());
        }
    } else {
        Matrix<CoefficientPrecision> transposed_A(A.nb_cols(), A.nb_rows());
        transpose(A, transposed_A);
        if (transb == 'C') {
            conj_if_complex(C.data(), C.nb_rows() * C.nb_cols());
        }
        if ((transa == 'T' && transb == 'C') or (transa == 'C' and transb != 'C')) {
            conj_if_complex(transposed_A.data(), transposed_A.nb_rows() * transposed_A.nb_cols());
        }
        sequential_internal_add_hmatrix_matrix_product_row_major(new_transa, 'N', transb == 'C' ? conj_if_complex(alpha) : alpha, B, transposed_A.data(), transb == 'C' ? conj_if_complex(beta) : beta, C.data(), C.nb_rows());
        if (transb == 'C') {
            conj_if_complex(C.data(), C.nb_rows() * C.nb_cols());
        }
    }
}

template <typename CoefficientPrecision, typename CoordinatePrecision = underlying_type<CoefficientPrecision>>
void internal_add_matrix_hmatrix_product(char transa, char transb, CoefficientPrecision alpha, const Matrix<CoefficientPrecision> &A, const HMatrix<CoefficientPrecision, CoordinatePrecision> &B, CoefficientPrecision beta, LowRankMatrix<CoefficientPrecision> &C) {
    bool C_is_overwritten = (beta == CoefficientPrecision(0) || C.rank_of() == 0);

    int nb_rows = (transa == 'N') ? A.nb_rows() : A.nb_cols();
    int nb_cols = (transb == 'N') ? B.nb_cols() : B.nb_rows();

    //
    Matrix<CoefficientPrecision> AB(nb_rows, nb_cols);
    internal_add_matrix_hmatrix_product(transa, transb, alpha, A, B, CoefficientPrecision(0), AB);

    // SVD truncation
    std::vector<underlying_type<CoefficientPrecision>> singular_values(std::min(nb_rows, nb_cols));
    Matrix<CoefficientPrecision> u(nb_rows, nb_rows);
    Matrix<CoefficientPrecision> vt(nb_cols, nb_cols);
    int truncated_rank = SVD_truncation(AB, C.get_epsilon(), u, vt, singular_values);

    // new_U=u*sqrt(tildeS) and new_V=sqrt(tildeS)*vt in the right dimensions
    Matrix<CoefficientPrecision> *new_U_ptr, *new_V_ptr;
    Matrix<CoefficientPrecision> U_1, V_1;
    if (C_is_overwritten) {
        new_U_ptr = &C.get_U();
        new_V_ptr = &C.get_V();
    } else {
        new_U_ptr = &U_1;
        new_V_ptr = &V_1;
    }

    {
        Matrix<CoefficientPrecision> &new_U = *new_U_ptr;
        Matrix<CoefficientPrecision> &new_V = *new_V_ptr;
        int M                               = nb_rows;
        int N                               = nb_cols;
        int incx                            = 1;
        new_U.resize(M, truncated_rank);
        new_V.resize(truncated_rank, N);
        CoefficientPrecision scaling_coef;
        for (int r = 0; r < truncated_rank; r++) {
            scaling_coef = std::sqrt(singular_values[r]);
            std::copy_n(u.data() + r * u.nb_rows(), u.nb_cols(), new_U.data() + r * M);
            Blas<CoefficientPrecision>::scal(&M, &scaling_coef, new_U.data() + r * M, &incx);
        }
        for (int r = 0; r < vt.nb_cols(); r++) {
            std::copy_n(vt.data() + r * vt.nb_rows(), truncated_rank, new_V.data() + r * truncated_rank);
        }

        for (int r = 0; r < truncated_rank; r++) {
            for (int j = 0; j < new_V.nb_cols(); j++) {
                new_V(r, j) = std::sqrt(singular_values[r]) * new_V(r, j);
            }
        }
    }

    if (C_is_overwritten) {
        return;
    }

    // Concatenate U_1 and U_2
    Matrix<CoefficientPrecision> &U_2 = C.get_U();
    Matrix<CoefficientPrecision> new_U(U_1.nb_rows(), U_1.nb_cols() + U_2.nb_cols());
    std::copy_n(U_1.data(), U_1.nb_rows() * U_1.nb_cols(), new_U.data());
    std::copy_n(U_2.data(), U_2.nb_rows() * U_2.nb_cols(), new_U.data() + U_1.nb_rows() * U_1.nb_cols());

    // Concatenate V_1 and V_2
    Matrix<CoefficientPrecision> &V_2 = C.get_V();
    scale(beta, V_2);
    Matrix<CoefficientPrecision> new_V(V_1.nb_rows() + V_2.nb_rows(), V_2.nb_cols());
    for (int j = 0; j < new_V.nb_cols(); j++) {
        std::copy_n(V_1.data() + j * V_1.nb_rows(), V_1.nb_rows(), new_V.data() + j * new_V.nb_rows());
        std::copy_n(V_2.data() + j * V_2.nb_rows(), V_2.nb_rows(), new_V.data() + j * new_V.nb_rows() + V_1.nb_rows());
    }

    // Set C
    C.get_U() = new_U;
    C.get_V() = new_V;
    SVD_recompression(C);
}
} // namespace htool

#endif
