
#ifndef HTOOL_TESTING_LOCAL_HMATRIX_HPP
#define HTOOL_TESTING_LOCAL_HMATRIX_HPP

#include "../basic_types/matrix.hpp"
#include "../clustering/cluster_node.hpp"
#include "../hmatrix/hmatrix.hpp"
#include "../hmatrix/interfaces/virtual_generator.hpp"
#include "../hmatrix/tree_builder/tree_builder.hpp"
#include "local_operator.hpp"

namespace htool {

template <typename CoefficientPrecision, typename CoordinatePrecision = underlying_type<CoefficientPrecision>>
class LocalHMatrix : public LocalOperator<CoefficientPrecision, CoordinatePrecision> {
    std::unique_ptr<HMatrix<CoefficientPrecision, CoordinatePrecision>> m_data{nullptr};

  public:
    LocalHMatrix(const VirtualGenerator<CoefficientPrecision> &mat, const HMatrixTreeBuilder<CoefficientPrecision, CoordinatePrecision> &hmatrix_tree_builder, std::shared_ptr<const Cluster<CoordinatePrecision>> cluster_tree_target, std::shared_ptr<const Cluster<CoordinatePrecision>> cluster_tree_source, char symmetry = 'N', char UPLO = 'N', bool target_use_permutation_to_mvprod = false, bool source_use_permutation_to_mvprod = false) : LocalOperator<CoefficientPrecision, CoordinatePrecision>(cluster_tree_target, cluster_tree_source, symmetry, UPLO, target_use_permutation_to_mvprod, source_use_permutation_to_mvprod) {
        m_data = std::make_unique<HMatrix<CoefficientPrecision, CoordinatePrecision>>(hmatrix_tree_builder.build(mat));
    }

    LocalHMatrix(const VirtualGenerator<CoefficientPrecision> &mat, std::shared_ptr<const Cluster<CoordinatePrecision>> cluster_tree_target, std::shared_ptr<const Cluster<CoordinatePrecision>> cluster_tree_source, htool::underlying_type<CoefficientPrecision> epsilon, CoordinatePrecision eta, char symmetry = 'N', char UPLO = 'N', bool target_use_permutation_to_mvprod = false, bool source_use_permutation_to_mvprod = false) : LocalHMatrix<CoefficientPrecision, CoordinatePrecision>(mat, HMatrixTreeBuilder<CoefficientPrecision, CoordinatePrecision>(cluster_tree_target, cluster_tree_source, epsilon, eta, symmetry, UPLO), cluster_tree_target, cluster_tree_source, symmetry, UPLO, target_use_permutation_to_mvprod, source_use_permutation_to_mvprod) {}

    void local_add_vector_product(char trans, CoefficientPrecision alpha, const CoefficientPrecision *in, CoefficientPrecision beta, CoefficientPrecision *out) const override { m_data->add_vector_product(trans, alpha, in, beta, out); }
    void local_add_vector_product_symmetric(char trans, CoefficientPrecision alpha, const CoefficientPrecision *in, CoefficientPrecision beta, CoefficientPrecision *out, char, char) const override { m_data->add_vector_product(trans, alpha, in, beta, out); }
    void local_add_matrix_product_row_major(char trans, CoefficientPrecision alpha, const CoefficientPrecision *in, CoefficientPrecision beta, CoefficientPrecision *out, int mu) const override { m_data->add_matrix_product_row_major(trans, alpha, in, beta, out, mu); }
    void local_add_matrix_product_symmetric_row_major(char trans, CoefficientPrecision alpha, const CoefficientPrecision *in, CoefficientPrecision beta, CoefficientPrecision *out, int mu, char, char) const override { m_data->add_matrix_product_row_major(trans, alpha, in, beta, out, mu); }

    const HMatrix<CoefficientPrecision, CoordinatePrecision> &get_hmatrix() const { return *m_data.get(); }
};
} // namespace htool
#endif
