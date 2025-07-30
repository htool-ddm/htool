
#ifndef HTOOL_DISTRIBUTED_OPERATOR_LOCAL_HMATRIX_HPP
#define HTOOL_DISTRIBUTED_OPERATOR_LOCAL_HMATRIX_HPP

#include "../../../clustering/cluster_node.hpp" // for Cluster
#include "../../../hmatrix/hmatrix.hpp"
#include "../../../hmatrix/linalg/add_hmatrix_matrix_product_row_major.hpp"
#include "../../../hmatrix/linalg/add_hmatrix_vector_product.hpp"
#include "../../../misc/misc.hpp"
#include "restricted_operator.hpp"

namespace htool {

template <typename CoefficientPrecision, typename CoordinatePrecision = underlying_type<CoefficientPrecision>>
class RestrictedGlobalToLocalHMatrix final : public RestrictedGlobalToLocalOperator<CoefficientPrecision> {
    const HMatrix<CoefficientPrecision, CoordinatePrecision> &m_data;

  public:
    RestrictedGlobalToLocalHMatrix(const HMatrix<CoefficientPrecision, CoordinatePrecision> &hmatrix, const LocalRenumbering &target_local_numbering, const LocalRenumbering &source_local_numbering, bool target_use_permutation_to_mvprod = false, bool source_use_permutation_to_mvprod = false) : RestrictedGlobalToLocalOperator<CoefficientPrecision>(target_local_numbering, source_local_numbering, target_use_permutation_to_mvprod, source_use_permutation_to_mvprod), m_data(hmatrix) {}

    RestrictedGlobalToLocalHMatrix(const RestrictedGlobalToLocalHMatrix &)                                              = default;
    RestrictedGlobalToLocalHMatrix &operator=(const RestrictedGlobalToLocalHMatrix &)                                   = default;
    RestrictedGlobalToLocalHMatrix(RestrictedGlobalToLocalHMatrix &&RestrictedGlobalToLocalHMatrix) noexcept            = default;
    RestrictedGlobalToLocalHMatrix &operator=(RestrictedGlobalToLocalHMatrix &&RestrictedGlobalToLocalHMatrix) noexcept = default;
    ~RestrictedGlobalToLocalHMatrix()                                                                                   = default;

    void local_add_vector_product(char trans, CoefficientPrecision alpha, const CoefficientPrecision *in, CoefficientPrecision beta, CoefficientPrecision *out) const override {
        openmp_internal_add_hmatrix_vector_product(trans, alpha, m_data, in, beta, out);
    }
    void local_add_matrix_product_row_major(char trans, CoefficientPrecision alpha, const CoefficientPrecision *in, CoefficientPrecision beta, CoefficientPrecision *out, int mu) const override {
        openmp_internal_add_hmatrix_matrix_product_row_major(trans, 'N', alpha, m_data, in, beta, out, mu);
    }

    const HMatrix<CoefficientPrecision, CoordinatePrecision> &get_hmatrix() const { return *m_data.get(); }
};
} // namespace htool
#endif
