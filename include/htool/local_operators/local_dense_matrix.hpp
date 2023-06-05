
#ifndef HTOOL_TESTING_LOCAL_DENSE_MATRIX_HPP
#define HTOOL_TESTING_LOCAL_DENSE_MATRIX_HPP

#include "../basic_types/matrix.hpp"
#include "../clustering/cluster_node.hpp"
#include "../hmatrix/interfaces/virtual_generator.hpp"
#include "local_operator.hpp"

namespace htool {

template <typename CoefficientPrecision, typename CoordinatePrecision = CoefficientPrecision>
class LocalDenseMatrix : public LocalOperator<CoefficientPrecision, CoordinatePrecision> {
    Matrix<CoefficientPrecision> m_data;

  public:
    LocalDenseMatrix(const VirtualGenerator<CoefficientPrecision> &mat, std::shared_ptr<const Cluster<CoordinatePrecision>> cluster_tree_target, std::shared_ptr<const Cluster<CoordinatePrecision>> cluster_tree_source, char symmetry = 'N', char UPLO = 'N', bool target_use_permutation_to_mvprod = false, bool source_use_permutation_to_mvprod = false) : LocalOperator<CoefficientPrecision, CoordinatePrecision>(cluster_tree_target, cluster_tree_source, symmetry, UPLO, target_use_permutation_to_mvprod, source_use_permutation_to_mvprod), m_data(cluster_tree_target->get_size(), cluster_tree_source->get_size()) {

        if (this->m_symmetry == 'N') {
            mat.copy_submatrix(m_data.nb_rows(), m_data.nb_cols(), this->m_target_root_cluster->get_offset(), this->m_source_root_cluster->get_offset(), m_data.data());
        } else if ((this->m_symmetry == 'S' || this->m_symmetry == 'H') && this->m_UPLO == 'L') {
            for (int i = 0; i < m_data.nb_rows(); i++) {
                for (int j = 0; j < i + 1; j++) {
                    mat.copy_submatrix(1, 1, i + this->m_target_root_cluster->get_offset(), j + this->m_source_root_cluster->get_offset(), m_data.data() + i + j * m_data.nb_rows());
                }
            }
        } else if ((this->m_symmetry == 'S' || this->m_symmetry == 'H') && this->m_UPLO == 'U') {
            for (int j = 0; j < m_data.nb_cols(); j++) {
                for (int i = 0; i < j + 1; i++) {
                    mat.copy_submatrix(1, 1, i + this->m_target_root_cluster->get_offset(), j + this->m_source_root_cluster->get_offset(), m_data.data() + i + j * m_data.nb_rows());
                }
            }
        }
    }

    void local_add_vector_product(char trans, CoefficientPrecision alpha, const CoefficientPrecision *in, CoefficientPrecision beta, CoefficientPrecision *out) const override { m_data.add_vector_product(trans, alpha, in, beta, out); }
    void local_add_vector_product_symmetric(char trans, CoefficientPrecision alpha, const CoefficientPrecision *in, CoefficientPrecision beta, CoefficientPrecision *out, char UPLO, char symmetry) const override { m_data.add_vector_product_symmetric(trans, alpha, in, beta, out, UPLO, symmetry); }
    void local_add_matrix_product_row_major(char trans, CoefficientPrecision alpha, const CoefficientPrecision *in, CoefficientPrecision beta, CoefficientPrecision *out, int mu) const override { m_data.add_matrix_product_row_major(trans, alpha, in, beta, out, mu); }
    void local_add_matrix_product_symmetric_row_major(char trans, CoefficientPrecision alpha, const CoefficientPrecision *in, CoefficientPrecision beta, CoefficientPrecision *out, int mu, char UPLO, char symmetry) const override { m_data.add_matrix_product_symmetric_row_major(trans, alpha, in, beta, out, mu, UPLO, symmetry); }
};
} // namespace htool
#endif
