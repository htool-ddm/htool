#ifndef HTOOL_LRMAT_LINALG_ADD_MATRIX_LRMAT_PRODUCT_HPP
#define HTOOL_LRMAT_LINALG_ADD_MATRIX_LRMAT_PRODUCT_HPP
#include "../../../matrix/linalg/add_matrix_matrix_product.hpp"
#include "../lrmat.hpp"
#include "../utils/recompression.hpp"

namespace htool {

template <typename CoefficientPrecision>
void add_matrix_lrmat_product(char transa, char transb, CoefficientPrecision alpha, const Matrix<CoefficientPrecision> &A, const LowRankMatrix<CoefficientPrecision> &B, CoefficientPrecision beta, Matrix<CoefficientPrecision> &C) {
    if (transb != 'N') {
        htool::Logger::get_instance().log(LogLevel::ERROR, "Operation is not implemented for add_matrix_lrmat_product (transb=" + std::string(1, transb) + ")"); // LCOV_EXCL_LINE
    }

    auto rank = B.rank_of();

    if (rank != 0) {
        auto &U = B.get_U();
        auto &V = B.get_V();
        Matrix<CoefficientPrecision> AU;
        if (transa == 'N') {
            AU.resize(A.nb_rows(), U.nb_cols());
        } else {
            AU.resize(A.nb_cols(), U.nb_cols());
        }
        add_matrix_matrix_product<CoefficientPrecision>(transa, 'N', 1, A, U, 0, AU);
        add_matrix_matrix_product<CoefficientPrecision>('N', 'N', alpha, AU, V, beta, C);
    }
}

template <typename CoefficientPrecision>
void add_matrix_lrmat_product(char transa, char transb, CoefficientPrecision alpha, const Matrix<CoefficientPrecision> &A, const LowRankMatrix<CoefficientPrecision> &B, CoefficientPrecision beta, LowRankMatrix<CoefficientPrecision> &C) {
    if (transb != 'N') {
        htool::Logger::get_instance().log(LogLevel::ERROR, "Operation is not implemented for add_matrix_lrmat_product (transb=" + std::string(1, transb) + ")"); // LCOV_EXCL_LINE
    }

    auto rank = B.rank_of();
    if (rank != 0) {
        auto &U_B = B.get_U();
        auto &V_B = B.get_V();
        auto &U_C = C.get_U();
        auto &V_C = C.get_V();
        if (beta == CoefficientPrecision(0) || C.rank_of() == 0) {
            if (transa == 'N') {
                U_C.resize(A.nb_rows(), U_B.nb_cols());
            } else {
                U_C.resize(A.nb_cols(), U_B.nb_cols());
            }
            V_C = V_B;
            add_matrix_matrix_product<CoefficientPrecision>(transa, 'N', alpha, A, U_B, 0, U_C);
        } else {

            // Concatenate V_B and V_C
            Matrix<CoefficientPrecision> new_V;
            scale(beta, V_C);
            new_V.resize(V_B.nb_rows() + V_C.nb_rows(), V_C.nb_cols());
            for (int j = 0; j < new_V.nb_cols(); j++) {
                std::copy_n(V_B.data() + j * V_B.nb_rows(), V_B.nb_rows(), new_V.data() + j * new_V.nb_rows());
                std::copy_n(V_C.data() + j * V_C.nb_rows(), V_C.nb_rows(), new_V.data() + j * new_V.nb_rows() + V_B.nb_rows());
            }

            // Compute AU= A*U_A
            Matrix<CoefficientPrecision> AU;
            if (transa == 'N') {
                AU.resize(A.nb_rows(), U_B.nb_cols());
            } else {
                AU.resize(A.nb_cols(), U_B.nb_cols());
            }
            add_matrix_matrix_product<CoefficientPrecision>(transa, 'N', alpha, A, U_B, 0, AU);

            // Concatenate U_A and U_C
            Matrix<CoefficientPrecision> new_U;
            new_U.resize(AU.nb_rows(), AU.nb_cols() + U_C.nb_cols());
            std::copy_n(AU.data(), AU.nb_rows() * AU.nb_cols(), new_U.data());
            std::copy_n(U_C.data(), U_C.nb_rows() * U_C.nb_cols(), new_U.data() + AU.nb_rows() * AU.nb_cols());

            C.get_U() = new_U;
            C.get_V() = new_V;
            recompression(C);
        }
    }
}

} // namespace htool

#endif
