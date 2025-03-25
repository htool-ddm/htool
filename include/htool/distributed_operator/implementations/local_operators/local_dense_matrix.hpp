
#ifndef HTOOL_DISTRIBUTED_OPERATOR_LOCAL_DENSE_MATRIX_HPP
#define HTOOL_DISTRIBUTED_OPERATOR_LOCAL_DENSE_MATRIX_HPP

#include "../../../clustering/cluster_node.hpp" // for Cluster
#include "../../../matrix/matrix.hpp"           // for Matrix
#include "local_operator.hpp"                   // for LocalOperator

namespace htool {

template <typename CoefficientPrecision, typename CoordinatePrecision = underlying_type<CoefficientPrecision>>
class LocalDenseMatrix final : public LocalOperator<CoefficientPrecision, CoordinatePrecision> {
    const Matrix<CoefficientPrecision> &m_data;

  public:
    LocalDenseMatrix(const Matrix<CoefficientPrecision> &matrix, const Cluster<CoordinatePrecision> &cluster_tree_target, const Cluster<CoordinatePrecision> &cluster_tree_source, char symmetry = 'N', char UPLO = 'N', bool target_use_permutation_to_mvprod = false, bool source_use_permutation_to_mvprod = false) : LocalOperator<CoefficientPrecision, CoordinatePrecision>(cluster_tree_target, cluster_tree_source, symmetry, UPLO, target_use_permutation_to_mvprod, source_use_permutation_to_mvprod), m_data(matrix) {}

    LocalDenseMatrix(const LocalDenseMatrix &)                                = default;
    LocalDenseMatrix &operator=(const LocalDenseMatrix &)                     = default;
    LocalDenseMatrix(LocalDenseMatrix &&LocalDenseMatrix) noexcept            = default;
    LocalDenseMatrix &operator=(LocalDenseMatrix &&LocalDenseMatrix) noexcept = default;
    ~LocalDenseMatrix()                                                       = default;

    void local_add_vector_product(char trans, CoefficientPrecision alpha, const CoefficientPrecision *in, CoefficientPrecision beta, CoefficientPrecision *out) const override {
        add_matrix_vector_product(trans, alpha, m_data, in, beta, out);
    }
    void local_add_vector_product_symmetric(char, CoefficientPrecision alpha, const CoefficientPrecision *in, CoefficientPrecision beta, CoefficientPrecision *out, char UPLO, char symmetry) const override {
        if (symmetry == 'S') {
            add_symmetric_matrix_vector_product(UPLO, alpha, m_data, in, beta, out);
        } else if (symmetry == 'H') {
            add_hermitian_matrix_vector_product(UPLO, alpha, m_data, in, beta, out);
        }
    }
    void local_add_matrix_product_row_major(char trans, CoefficientPrecision alpha, const CoefficientPrecision *in, CoefficientPrecision beta, CoefficientPrecision *out, int mu) const override {
        add_matrix_matrix_product_row_major(trans, 'N', alpha, m_data, in, beta, out, mu);
    }
    void local_add_matrix_product_symmetric_row_major(char, CoefficientPrecision alpha, const CoefficientPrecision *in, CoefficientPrecision beta, CoefficientPrecision *out, int mu, char UPLO, char symmetry) const override {
        if (symmetry == 'S') {
            add_symmetric_matrix_matrix_product_row_major('L', UPLO, alpha, m_data, in, beta, out, mu);
        } else if (symmetry == 'H') {
            add_hermitian_matrix_matrix_product_row_major('L', UPLO, alpha, m_data, in, beta, out, mu);
        }
    }
};
} // namespace htool
#endif
