#ifndef HTOOL_LRMAT_LINALG_ADD_MATRIX_LRMAT_PRODUCT_HPP
#define HTOOL_LRMAT_LINALG_ADD_MATRIX_LRMAT_PRODUCT_HPP

#include "../../../matrix/linalg/add_matrix_matrix_product.hpp" // for add_...
#include "../../../matrix/linalg/scale.hpp"                     // for scale
#include "../../../matrix/linalg/transpose.hpp"                 // for tran...
#include "../../../matrix/matrix.hpp"                           // for Matrix
#include "../../../misc/misc.hpp"                               // for conj...
#include "../lrmat.hpp"                                         // for LowR...
#include "../utils/SVD_recompression.hpp"                       // for reco...
#include <algorithm>                                            // for copy_n

namespace htool {

template <typename CoefficientPrecision>
void add_matrix_lrmat_product(char transa, char transb, CoefficientPrecision alpha, const Matrix<CoefficientPrecision> &A, const LowRankMatrix<CoefficientPrecision> &B, CoefficientPrecision beta, Matrix<CoefficientPrecision> &C) {
    auto rank = B.rank_of();

    if (rank != 0) {
        auto &U = B.get_U();
        auto &V = B.get_V();
        Matrix<CoefficientPrecision> AU;
        if (transa == 'N') {
            AU.resize(A.nb_rows(), transb == 'N' ? U.nb_cols() : V.nb_rows());
        } else {
            AU.resize(A.nb_cols(), transb == 'N' ? U.nb_cols() : V.nb_rows());
        }
        if (transb == 'N') {
            add_matrix_matrix_product<CoefficientPrecision>(transa, 'N', 1, A, U, 0, AU);
            add_matrix_matrix_product<CoefficientPrecision>('N', 'N', alpha, AU, V, beta, C);
        } else {
            add_matrix_matrix_product<CoefficientPrecision>(transa, transb, 1, A, V, 0, AU);
            add_matrix_matrix_product<CoefficientPrecision>('N', transb, alpha, AU, U, beta, C);
        }
    }
}

template <typename CoefficientPrecision>
void add_matrix_lrmat_product(char transa, char transb, CoefficientPrecision alpha, const Matrix<CoefficientPrecision> &A, const LowRankMatrix<CoefficientPrecision> &B, CoefficientPrecision beta, LowRankMatrix<CoefficientPrecision> &C) {
    auto rank = B.rank_of();
    if (rank != 0) {
        auto &U_B = B.get_U();
        auto &V_B = B.get_V();
        auto &U_C = C.get_U();
        auto &V_C = C.get_V();
        if (beta == CoefficientPrecision(0) || C.rank_of() == 0) {
            if (transa == 'N') {
                U_C.resize(A.nb_rows(), transb == 'N' ? U_B.nb_cols() : V_B.nb_rows());
            } else {
                U_C.resize(A.nb_cols(), transb == 'N' ? U_B.nb_cols() : V_B.nb_rows());
            }
            if (transb == 'N') {
                V_C = V_B;
                add_matrix_matrix_product<CoefficientPrecision>(transa, transb, alpha, A, U_B, 0, U_C);
            } else {
                V_C.resize(U_B.nb_cols(), U_B.nb_rows());
                transpose(U_B, V_C);
                if (transb == 'C') {
                    conj_if_complex(V_C.data(), V_C.nb_rows() * V_C.nb_cols());
                }
                add_matrix_matrix_product<CoefficientPrecision>(transa, transb, alpha, A, V_B, 0, U_C);
            }
        } else {

            // Concatenate V_B and V_C
            Matrix<CoefficientPrecision> new_V;
            scale(beta, V_C);
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

            // Compute AU= A*U_A
            Matrix<CoefficientPrecision> AU;
            if (transa == 'N') {
                AU.resize(A.nb_rows(), transb == 'N' ? U_B.nb_cols() : V_B.nb_rows());
            } else {
                AU.resize(A.nb_cols(), transb == 'N' ? U_B.nb_cols() : V_B.nb_rows());
            }
            if (transb == 'N') {
                add_matrix_matrix_product<CoefficientPrecision>(transa, 'N', alpha, A, U_B, 0, AU);
            } else {
                add_matrix_matrix_product<CoefficientPrecision>(transa, transb, alpha, A, V_B, 0, AU);
            }

            // Concatenate U_A and U_C
            Matrix<CoefficientPrecision> new_U;
            new_U.resize(AU.nb_rows(), AU.nb_cols() + U_C.nb_cols());
            std::copy_n(AU.data(), AU.nb_rows() * AU.nb_cols(), new_U.data());
            std::copy_n(U_C.data(), U_C.nb_rows() * U_C.nb_cols(), new_U.data() + AU.nb_rows() * AU.nb_cols());

            C.get_U() = new_U;
            C.get_V() = new_V;
            SVD_recompression(C);
        }
    }
}

} // namespace htool

#endif
