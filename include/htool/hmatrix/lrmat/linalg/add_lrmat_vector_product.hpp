#ifndef HTOOL_LRMAT_LINALG_ADD_LRMAT_VECTOR_PRODUCT_HPP
#define HTOOL_LRMAT_LINALG_ADD_LRMAT_VECTOR_PRODUCT_HPP
#include "../../../matrix/linalg/add_matrix_vector_product.hpp"
#include "../lrmat.hpp"
namespace htool {

template <typename CoefficientPrecision>
void add_lrmat_vector_product(char trans, CoefficientPrecision alpha, const LowRankMatrix<CoefficientPrecision> &A, const CoefficientPrecision *in, CoefficientPrecision beta, CoefficientPrecision *out) {
    auto rank = A.rank_of();
    if (rank != 0) {
        auto &U = A.get_U();
        auto &V = A.get_V();
        if (trans == 'N') {
            std::vector<CoefficientPrecision> a(rank);
            add_matrix_vector_product<CoefficientPrecision>(trans, CoefficientPrecision(1.), V, in, 0, a.data());
            add_matrix_vector_product<CoefficientPrecision>(trans, alpha, U, a.data(), beta, out);
        } else {
            std::vector<CoefficientPrecision> a(rank);
            add_matrix_vector_product<CoefficientPrecision>(trans, CoefficientPrecision(1.), U, in, 0, a.data());
            add_matrix_vector_product<CoefficientPrecision>(trans, alpha, V, a.data(), beta, out);
        }
    }
}

} // namespace htool

#endif
