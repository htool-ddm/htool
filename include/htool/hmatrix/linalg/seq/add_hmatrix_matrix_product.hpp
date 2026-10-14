#ifndef HTOOL_HMATRIX_LINALG_SEQ_ADD_HMATRIX_MATRIX_PRODUCT_HPP
#define HTOOL_HMATRIX_LINALG_SEQ_ADD_HMATRIX_MATRIX_PRODUCT_HPP

#include "../../../matrix/linalg/scale.hpp"
#include "../../../matrix/linalg/transpose.hpp"
#include "../../../matrix/matrix.hpp"
#include "../../../matrix/utils/SVD_truncation.hpp"
#include "../../../misc/misc.hpp"
#include "../../../wrappers/wrapper_blas.hpp"
#include "../../hmatrix.hpp"
#include "../../lrmat/lrmat.hpp"
#include "../../lrmat/utils/SVD_recompression.hpp"
#include "add_hmatrix_matrix_product_row_major.hpp"
#include <algorithm>
#include <vector>

namespace htool {

template <typename MatB,
          typename MatC,
          typename CoordinatePrecision = underlying_type<typename MatB::value_type>,
          typename                     = std::enable_if_t<
              std::is_same<typename MatB::value_type, typename MatC::value_type>::value>>
void internal_add_hmatrix_matrix_product(char transa, char transb, typename MatB::value_type alpha, const HMatrix<typename MatB::value_type, CoordinatePrecision> &A, const MatB &B, typename MatB::value_type beta, MatC &C) {
    using CoefficientPrecision = typename MatB::value_type;
    if (transb == 'N') {
        Matrix<CoefficientPrecision> transposed_B(B.nb_cols(), B.nb_rows()), transposed_C(C.nb_cols(), C.nb_rows());
        transpose(B, transposed_B);
        transpose(C, transposed_C);
        sequential_internal_add_hmatrix_matrix_product_row_major(transa, transb, alpha, A, transposed_B.data(), beta, transposed_C.data(), transposed_C.nb_rows());
        transpose(transposed_C, C);
    } else {
        Matrix<CoefficientPrecision> transposed_C(C.nb_cols(), C.nb_rows());
        transpose(C, transposed_C);
        std::vector<CoefficientPrecision> buffer_B(transb == 'C' ? B.nb_cols() * B.nb_rows() : 0);

        if (transb == 'C') {
            std::copy(B.data(), B.data() + B.nb_cols() * B.nb_rows(), buffer_B.data());
            conj_if_complex(buffer_B.data(), buffer_B.size());
        }

        sequential_internal_add_hmatrix_matrix_product_row_major(transa, 'N', alpha, A, transb == 'C' ? buffer_B.data() : B.data(), beta, transposed_C.data(), transposed_C.nb_rows());
        transpose(transposed_C, C);
    }
}

template <typename CoefficientPrecision, typename CoordinatePrecision = underlying_type<CoefficientPrecision>>
void internal_add_hmatrix_matrix_product(char transa, char transb, CoefficientPrecision alpha, const HMatrix<CoefficientPrecision, CoordinatePrecision> &A, const Matrix<CoefficientPrecision> &B, CoefficientPrecision beta, LowRankMatrix<CoefficientPrecision> &C) {
    bool C_is_overwritten = (beta == CoefficientPrecision(0) || C.rank_of() == 0);

    int nb_rows = (transa == 'N') ? A.nb_rows() : A.nb_cols();
    int nb_cols = (transb == 'N') ? B.nb_cols() : B.nb_rows();

    //
    Matrix<CoefficientPrecision> AB(nb_rows, nb_cols);
    internal_add_hmatrix_matrix_product(transa, transb, alpha, A, B, CoefficientPrecision(0), AB);

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
