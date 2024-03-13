
#ifndef HTOOL_TESTING_LOCAL_HMATRIX_HPP
#define HTOOL_TESTING_LOCAL_HMATRIX_HPP

#include "../clustering/cluster_node.hpp"
#include "../hmatrix/hmatrix.hpp"
#include "../hmatrix/interfaces/virtual_generator.hpp"
#include "../hmatrix/tree_builder/tree_builder.hpp"
#include "../matrix/matrix.hpp"
#include "local_operator.hpp"

namespace htool {

template <typename CoefficientPrecision, typename CoordinatePrecision = underlying_type<CoefficientPrecision>>
class LocalHMatrix final : public LocalOperator<CoefficientPrecision, CoordinatePrecision> {
    const HMatrix<CoefficientPrecision, CoordinatePrecision> &m_data;

  public:
    LocalHMatrix(const HMatrix<CoefficientPrecision, CoordinatePrecision> &hmatrix, const Cluster<CoordinatePrecision> &cluster_tree_target, const Cluster<CoordinatePrecision> &cluster_tree_source, char symmetry = 'N', char UPLO = 'N', bool target_use_permutation_to_mvprod = false, bool source_use_permutation_to_mvprod = false) : LocalOperator<CoefficientPrecision, CoordinatePrecision>(cluster_tree_target, cluster_tree_source, symmetry, UPLO, target_use_permutation_to_mvprod, source_use_permutation_to_mvprod), m_data(hmatrix) {}

    LocalHMatrix(const LocalHMatrix &)                            = default;
    LocalHMatrix &operator=(const LocalHMatrix &)                 = default;
    LocalHMatrix(LocalHMatrix &&LocalHMatrix) noexcept            = default;
    LocalHMatrix &operator=(LocalHMatrix &&LocalHMatrix) noexcept = default;
    ~LocalHMatrix()                                               = default;

    void local_add_vector_product(char trans, CoefficientPrecision alpha, const CoefficientPrecision *in, CoefficientPrecision beta, CoefficientPrecision *out) const override { m_data.add_vector_product(trans, alpha, in, beta, out); }
    void local_add_vector_product_symmetric(char trans, CoefficientPrecision alpha, const CoefficientPrecision *in, CoefficientPrecision beta, CoefficientPrecision *out, char, char) const override { m_data.add_vector_product(trans, alpha, in, beta, out); }
    void local_add_matrix_product_row_major(char trans, CoefficientPrecision alpha, const CoefficientPrecision *in, CoefficientPrecision beta, CoefficientPrecision *out, int mu) const override { m_data.add_matrix_product_row_major(trans, alpha, in, beta, out, mu); }
    void local_add_matrix_product_symmetric_row_major(char trans, CoefficientPrecision alpha, const CoefficientPrecision *in, CoefficientPrecision beta, CoefficientPrecision *out, int mu, char, char) const override { m_data.add_matrix_product_row_major(trans, alpha, in, beta, out, mu); }

    const HMatrix<CoefficientPrecision, CoordinatePrecision> &get_hmatrix() const { return *m_data.get(); }
};
} // namespace htool
#endif
