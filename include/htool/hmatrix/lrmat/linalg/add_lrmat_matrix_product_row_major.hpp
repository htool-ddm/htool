#ifndef HTOOL_LRMAT_LINALG_ADD_LRMAT_MATRIX_PRODUCT_ROW_MAJOR_HPP
#define HTOOL_LRMAT_LINALG_ADD_LRMAT_MATRIX_PRODUCT_ROW_MAJOR_HPP
#include "../../../matrix/linalg/add_matrix_matrix_product_row_major.hpp"
#include "../lrmat.hpp"

namespace htool {

template <typename CoefficientPrecision>
void add_lrmat_matrix_product_row_major(char transa, char transb, CoefficientPrecision alpha, const LowRankMatrix<CoefficientPrecision> &A, const CoefficientPrecision *in, CoefficientPrecision beta, CoefficientPrecision *out, int mu) {
    if (transb != 'N') {
        htool::Logger::get_instance().log(LogLevel::ERROR, "Operation is not implemented for add_lrmat_matrix_product_row_major (transb=" + std::string(1, transb) + ")"); // LCOV_EXCL_LINE
    }

    auto rank = A.rank_of();

    if (rank != 0) {
        auto &U = A.get_U();
        auto &V = A.get_V();
        if (transa == 'N') {
            std::vector<CoefficientPrecision> a(rank * mu);
            V.add_matrix_product_row_major(transa, 1, in, 0, a.data(), mu);
            U.add_matrix_product_row_major(transa, alpha, a.data(), beta, out, mu);
        } else {
            std::vector<CoefficientPrecision> a(rank * mu);
            U.add_matrix_product_row_major(transa, 1, in, 0, a.data(), mu);
            V.add_matrix_product_row_major(transa, alpha, a.data(), beta, out, mu);
        }
    }
}

} // namespace htool

#endif
