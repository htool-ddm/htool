
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
    LocalHMatrix(const VirtualGenerator<CoefficientPrecision> &mat, std::shared_ptr<const Cluster<CoordinatePrecision>> cluster_tree_target, std::shared_ptr<const Cluster<CoordinatePrecision>> cluster_tree_source, htool::underlying_type<CoefficientPrecision> epsilon, CoordinatePrecision eta, char symmetry = 'N', char UPLO = 'N', bool target_use_permutation_to_mvprod = false, bool source_use_permutation_to_mvprod = false) : LocalOperator<CoefficientPrecision, CoordinatePrecision>(cluster_tree_target, cluster_tree_source, symmetry, UPLO, target_use_permutation_to_mvprod, source_use_permutation_to_mvprod) {

        HMatrixTreeBuilder<CoefficientPrecision, CoordinatePrecision> hmatrix_tree_builder(this->m_target_root_cluster, this->m_source_root_cluster, epsilon, eta, this->m_symmetry, this->m_UPLO);

        m_data = std::unique_ptr<HMatrix<CoefficientPrecision, CoordinatePrecision>>(new HMatrix<CoefficientPrecision, CoordinatePrecision>(hmatrix_tree_builder.build(mat)));
    }

    void local_add_vector_product(char trans, CoefficientPrecision alpha, const CoefficientPrecision *in, CoefficientPrecision beta, CoefficientPrecision *out) const override { m_data->add_vector_product(trans, alpha, in, beta, out); }
    void local_add_vector_product_symmetric(char trans, CoefficientPrecision alpha, const CoefficientPrecision *in, CoefficientPrecision beta, CoefficientPrecision *out, char UPLO, char symmetry) const override { m_data->add_vector_product(trans, alpha, in, beta, out); }
    void local_add_matrix_product_row_major(char trans, CoefficientPrecision alpha, const CoefficientPrecision *in, CoefficientPrecision beta, CoefficientPrecision *out, int mu) const override { m_data->add_matrix_product_row_major(trans, alpha, in, beta, out, mu); }
    void local_add_matrix_product_symmetric_row_major(char trans, CoefficientPrecision alpha, const CoefficientPrecision *in, CoefficientPrecision beta, CoefficientPrecision *out, int mu, char UPLO, char symmetry) const override { m_data->add_matrix_product_row_major(trans, alpha, in, beta, out, mu); }
};
} // namespace htool
#endif
