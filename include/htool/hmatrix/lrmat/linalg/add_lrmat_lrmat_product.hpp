#ifndef HTOOL_LRMAT_LINALG_ADD_LRMAT_LRMAT_PRODUCT_HPP
#define HTOOL_LRMAT_LINALG_ADD_LRMAT_LRMAT_PRODUCT_HPP
#include "../../../hmatrix/lrmat/utils/recompression.hpp"       // for reco...
#include "../../../matrix/linalg/add_matrix_matrix_product.hpp" // for add_...
#include "../../../matrix/linalg/scale.hpp"                     // for scale
#include "../../../matrix/linalg/transpose.hpp"                 // for tran...
#include "../../../matrix/matrix.hpp"                           // for Matrix
#include "../../../misc/misc.hpp"                               // for conj...
#include "../lrmat.hpp"                                         // for LowR...
#include "add_lrmat_matrix_product.hpp"                         // for add_...
#include <algorithm>                                            // for copy_n

namespace htool {

template <typename CoefficientPrecision>
void add_lrmat_lrmat_product(char transa, char transb, CoefficientPrecision alpha, const LowRankMatrix<CoefficientPrecision> &A, const LowRankMatrix<CoefficientPrecision> &B, CoefficientPrecision beta, Matrix<CoefficientPrecision> &C) {
    auto rank_A = A.rank_of();
    auto rank_B = B.rank_of();
    if (rank_A != 0 and rank_B != 0) { // the order of operations could be optimized
        auto &U_B = B.get_U();
        auto &V_B = B.get_V();
        if (transb == 'N') {
            Matrix<CoefficientPrecision> AUB(C.nb_rows(), B.rank_of());
            add_lrmat_matrix_product(transa, 'N', CoefficientPrecision(1), A, U_B, CoefficientPrecision(0), AUB);
            add_matrix_matrix_product('N', 'N', alpha, AUB, V_B, beta, C);
        } else {
            Matrix<CoefficientPrecision> AVBt(C.nb_rows(), V_B.nb_rows());
            add_lrmat_matrix_product(transa, transb, CoefficientPrecision(1), A, V_B, CoefficientPrecision(0), AVBt);
            add_matrix_matrix_product('N', transb, alpha, AVBt, U_B, beta, C);
        }
    }
}

template <typename CoefficientPrecision>
void add_lrmat_lrmat_product(char transa, char transb, CoefficientPrecision alpha, const LowRankMatrix<CoefficientPrecision> &A, const LowRankMatrix<CoefficientPrecision> &B, CoefficientPrecision beta, LowRankMatrix<CoefficientPrecision> &C) {
    auto rank_A = A.rank_of();
    auto rank_B = B.rank_of();
    if (rank_A != 0 and rank_B != 0) { // the order of operations could be optimized
        auto &U_B = B.get_U();
        auto &V_B = B.get_V();
        auto &U_C = C.get_U();
        auto &V_C = C.get_V();
        if (beta == CoefficientPrecision(0) || C.rank_of() == 0) {
            U_C.resize(transa == 'N' ? A.nb_rows() : A.nb_cols(), transb == 'N' ? U_B.nb_cols() : V_B.nb_rows());
            if (transb == 'N') {
                V_C = V_B;
                add_lrmat_matrix_product(transa, 'N', alpha, A, U_B, CoefficientPrecision(0), U_C);
            } else {
                V_C.resize(U_B.nb_cols(), U_B.nb_rows());
                transpose(U_B, V_C);
                if (transb == 'C') {
                    conj_if_complex(V_C.data(), V_C.nb_rows() * V_C.nb_cols());
                }
                add_lrmat_matrix_product(transa, transb, alpha, A, V_B, CoefficientPrecision(0), U_C);
            }
        } else {
            Matrix<CoefficientPrecision> new_U, sub_new_U;
            if (transb == 'N') {
                new_U.resize(C.nb_rows(), U_B.nb_cols() + U_C.nb_cols());
            } else {
                new_U.resize(C.nb_rows(), V_B.nb_rows() + U_C.nb_cols());
            }
            sub_new_U.assign(C.nb_rows(), U_B.nb_cols(), new_U.data(), false);
            if (transb == 'N') {
                add_lrmat_matrix_product(transa, 'N', alpha, A, U_B, CoefficientPrecision(0), sub_new_U);
            } else {
                add_lrmat_matrix_product(transa, transb, alpha, A, V_B, CoefficientPrecision(0), sub_new_U);
            }
            std::copy_n(U_C.data(), U_C.nb_cols() * U_C.nb_rows(), new_U.data() + (C.nb_rows() * U_B.nb_cols()));

            // Concatenate V_B and V_C
            scale(beta, V_C);
            Matrix<CoefficientPrecision> new_V;

            if (transb == 'N') {
                new_V.resize(V_B.nb_rows() + V_C.nb_rows(), V_C.nb_cols());
                for (int j = 0; j < new_V.nb_cols(); j++) {
                    std::copy_n(V_B.data() + j * V_B.nb_rows(), V_B.nb_rows(), new_V.data() + j * new_V.nb_rows());
                    std::copy_n(V_C.data() + j * V_C.nb_rows(), V_C.nb_rows(), new_V.data() + j * new_V.nb_rows() + V_B.nb_rows());
                }
            } else {
                new_V.resize(U_B.nb_cols() + V_C.nb_rows(), V_C.nb_cols());
                Matrix<CoefficientPrecision> temp(U_B.nb_cols(), U_B.nb_rows());
                transpose(U_B, temp);
                if (transb == 'C') {
                    conj_if_complex(temp.data(), temp.nb_rows() * temp.nb_cols());
                }
                for (int j = 0; j < new_V.nb_cols(); j++) {
                    std::copy_n(temp.data() + j * temp.nb_rows(), temp.nb_rows(), new_V.data() + j * new_V.nb_rows());
                    std::copy_n(V_C.data() + j * V_C.nb_rows(), V_C.nb_rows(), new_V.data() + j * new_V.nb_rows() + V_B.nb_rows());
                }
            }

            C.get_U() = new_U;
            C.get_V() = new_V;
            recompression(C);
        }
    }
}

} // namespace htool

#endif
