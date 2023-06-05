#ifndef HTOOL_LRMAT_LINALG_ADD_LRMAT_MATRIX_PRODUCT_ROW_MAJOR_HPP
#define HTOOL_LRMAT_LINALG_ADD_LRMAT_MATRIX_PRODUCT_ROW_MAJOR_HPP

#include "../../../matrix/linalg/add_matrix_matrix_product_row_major.hpp"
#include "../lrmat.hpp"
#include <vector>

namespace htool {

template <typename CoefficientPrecision>
void add_lrmat_matrix_product_row_major(char transa, char transb, CoefficientPrecision alpha, const LowRankMatrix<CoefficientPrecision> &A, const CoefficientPrecision *in, CoefficientPrecision beta, CoefficientPrecision *out, int mu) {
    auto rank = A.rank_of();

    if (rank != 0) {
        auto &U = A.get_U();
        auto &V = A.get_V();
        if (transa == 'N') {
            std::vector<CoefficientPrecision> a(rank * mu);
            add_matrix_matrix_product_row_major(transa, transb, CoefficientPrecision(1), V, in, CoefficientPrecision(0), a.data(), mu);
            add_matrix_matrix_product_row_major(transa, 'N', alpha, U, a.data(), beta, out, mu);
        } else {
            std::vector<CoefficientPrecision> a(rank * mu);
            add_matrix_matrix_product_row_major(transa, transb, CoefficientPrecision(1), U, in, CoefficientPrecision(0), a.data(), mu);
            add_matrix_matrix_product_row_major(transa, 'N', alpha, V, a.data(), beta, out, mu);
        }
    }
}

} // namespace htool

#endif
