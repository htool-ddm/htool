#ifndef HTOOL_HMATRIX_LINALG_ADD_LOW_RANK_MATRIX_HMATRIX_PRODUCT_HPP
#define HTOOL_HMATRIX_LINALG_ADD_LOW_RANK_MATRIX_HMATRIX_PRODUCT_HPP

#include "../../matrix/linalg/scale.hpp"     // for scale
#include "../../matrix/linalg/transpose.hpp" // for transpose
#include "../../matrix/matrix.hpp"           // for Matrix
#include "../../misc/misc.hpp"               // for conj_if_complex
#include "../hmatrix.hpp"                    // for HMatrix
#include "../lrmat/lrmat.hpp"                // for LowRankMatrix
#include "../lrmat/utils/recompression.hpp"  // for recompression
#include "add_lrmat_hmatrix.hpp"             // for add_lrmat_hma...
#include "add_matrix_hmatrix_product.hpp"    // for add_matrix_hm...
#include <algorithm>                         // for copy_n

namespace htool {

template <typename CoefficientPrecision, typename CoordinatePrecision = CoefficientPrecision>
void internal_add_lrmat_hmatrix_product(char transa, char transb, CoefficientPrecision alpha, const LowRankMatrix<CoefficientPrecision> &A, const HMatrix<CoefficientPrecision, CoordinatePrecision> &B, CoefficientPrecision beta, Matrix<CoefficientPrecision> &C) {
    auto rank = A.rank_of();
    if (rank != 0) {
        auto &U = A.get_U();
        auto &V = A.get_V();
        if (transa == 'N') {
            Matrix<CoefficientPrecision> VB(V.nb_rows(), transb == 'N' ? B.nb_cols() : B.nb_rows());
            internal_add_matrix_hmatrix_product<CoefficientPrecision>(transa, transb, CoefficientPrecision(1), V, B, CoefficientPrecision(0), VB);
            add_matrix_matrix_product<CoefficientPrecision>(transa, 'N', alpha, U, VB, beta, C);
        } else {
            Matrix<CoefficientPrecision> UtB(V.nb_rows(), transb == 'N' ? B.nb_cols() : B.nb_rows());
            internal_add_matrix_hmatrix_product<CoefficientPrecision>(transa, transb, CoefficientPrecision(1), U, B, CoefficientPrecision(0), UtB);
            add_matrix_matrix_product<CoefficientPrecision>(transa, 'N', alpha, V, UtB, beta, C);
        }
    }
}

template <typename CoefficientPrecision, typename CoordinatePrecision = CoefficientPrecision>
void internal_add_lrmat_hmatrix_product(char transa, char transb, CoefficientPrecision alpha, const LowRankMatrix<CoefficientPrecision> &A, const HMatrix<CoefficientPrecision, CoordinatePrecision> &B, CoefficientPrecision beta, LowRankMatrix<CoefficientPrecision> &C) {
    auto rank = A.rank_of();
    if (rank != 0) {
        auto &U_A = A.get_U();
        auto &V_A = A.get_V();
        auto &U_C = C.get_U();
        auto &V_C = C.get_V();
        if (beta == CoefficientPrecision(0) || C.rank_of() == 0) {
            if (transa == 'N') {
                V_C.resize(V_A.nb_rows(), transb == 'N' ? B.nb_cols() : B.nb_rows());
                U_C = U_A;
                internal_add_matrix_hmatrix_product<CoefficientPrecision>(transa, transb, alpha, V_A, B, 0, V_C);
            } else {
                V_C.resize(U_A.nb_cols(), transb == 'N' ? B.nb_cols() : B.nb_rows());
                U_C.resize(V_A.nb_cols(), V_A.nb_rows());
                transpose(V_A, U_C);
                if (transa == 'C') {
                    conj_if_complex(U_C.data(), U_C.nb_rows() * U_C.nb_cols());
                }
                internal_add_matrix_hmatrix_product<CoefficientPrecision>(transa, transb, alpha, U_A, B, 0, V_C);
            }
        } else {
            Matrix<CoefficientPrecision> VB;
            Matrix<CoefficientPrecision> new_U;
            if (transa == 'N') {
                // Concatenate U_A and U_C
                new_U.resize(U_A.nb_rows(), U_A.nb_cols() + U_C.nb_cols());
                std::copy_n(U_A.data(), U_A.nb_rows() * U_A.nb_cols(), new_U.data());
                std::copy_n(U_C.data(), U_C.nb_rows() * U_C.nb_cols(), new_U.data() + U_A.nb_rows() * U_A.nb_cols());

                // Compute VB=V_B*B
                VB.resize(V_A.nb_rows(), transb == 'N' ? B.nb_cols() : B.nb_rows());
                internal_add_matrix_hmatrix_product<CoefficientPrecision>(transa, transb, alpha, V_A, B, 0, VB);
            } else {
                // Concatenate V_At and U_C
                new_U.resize(V_A.nb_cols(), U_A.nb_cols() + U_C.nb_cols());
                if (transa == 'T') {
                    for (int i = 0; i < V_A.nb_rows(); i++) {
                        for (int j = 0; j < V_A.nb_cols(); j++) {
                            new_U(j, i) = V_A(i, j);
                        }
                    }
                } else {
                    for (int i = 0; i < V_A.nb_rows(); i++) {
                        for (int j = 0; j < V_A.nb_cols(); j++) {
                            new_U(j, i) = conj_if_complex(V_A(i, j));
                        }
                    }
                }
                std::copy_n(U_C.data(), U_C.nb_rows() * U_C.nb_cols(), new_U.data() + V_A.nb_rows() * V_A.nb_cols());

                // Compute VB=V_B*B
                VB.resize(V_A.nb_rows(), transb == 'N' ? B.nb_cols() : B.nb_rows());
                internal_add_matrix_hmatrix_product<CoefficientPrecision>(transa, transb, alpha, U_A, B, 0, VB);
            }

            // Concatenate VB and V_C
            scale(beta, V_C);
            Matrix<CoefficientPrecision> new_V(VB.nb_rows() + V_C.nb_rows(), V_C.nb_cols());
            for (int j = 0; j < new_V.nb_cols(); j++) {
                std::copy_n(VB.data() + j * VB.nb_rows(), VB.nb_rows(), new_V.data() + j * new_V.nb_rows());
                std::copy_n(V_C.data() + j * V_C.nb_rows(), V_C.nb_rows(), new_V.data() + j * new_V.nb_rows() + VB.nb_rows());
            }
            C.get_U() = new_U;
            C.get_V() = new_V;
            recompression(C);
        }
    }
}

template <typename CoefficientPrecision, typename CoordinatePrecision = CoefficientPrecision>
void internal_add_lrmat_hmatrix_product(char transa, char transb, CoefficientPrecision alpha, const LowRankMatrix<CoefficientPrecision> &A, const HMatrix<CoefficientPrecision, CoordinatePrecision> &B, CoefficientPrecision beta, HMatrix<CoefficientPrecision, CoordinatePrecision> &C) {
    if (beta != CoefficientPrecision(1)) {
        scale(beta, C);
    }

    LowRankMatrix<CoefficientPrecision> lrmat(A.get_epsilon());
    internal_add_lrmat_hmatrix_product(transa, transb, alpha, A, B, CoefficientPrecision(1), lrmat);
    internal_add_lrmat_hmatrix(lrmat, C);
}

} // namespace htool

#endif
