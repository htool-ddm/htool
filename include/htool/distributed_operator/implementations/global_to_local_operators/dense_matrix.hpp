
#ifndef HTOOL_DISTRIBUTED_OPERATOR_LOCAL_DENSE_MATRIX_HPP
#define HTOOL_DISTRIBUTED_OPERATOR_LOCAL_DENSE_MATRIX_HPP

#include "../../../matrix/matrix.hpp" // for Matrix
#include "restricted_operator.hpp"    // for LocalOperator

namespace htool {

template <typename CoefficientPrecision>
class RestrictedGlobalToLocalDenseMatrix final : public RestrictedGlobalToLocalOperator<CoefficientPrecision> {
    const Matrix<CoefficientPrecision> &m_data;
    char m_symmetry = 'N';
    char m_UPLO     = 'N';

  public:
    RestrictedGlobalToLocalDenseMatrix(const Matrix<CoefficientPrecision> &matrix, const LocalRenumbering &target_local_renumbering, const LocalRenumbering &source_local_renumbering, char symmetry = 'N', char UPLO = 'N', bool target_use_permutation_to_mvprod = false, bool source_use_permutation_to_mvprod = false) : RestrictedGlobalToLocalOperator<CoefficientPrecision>(target_local_renumbering, source_local_renumbering, target_use_permutation_to_mvprod, source_use_permutation_to_mvprod), m_data(matrix), m_symmetry(symmetry), m_UPLO(UPLO) {}

    RestrictedGlobalToLocalDenseMatrix(const RestrictedGlobalToLocalDenseMatrix &)                                                  = default;
    RestrictedGlobalToLocalDenseMatrix &operator=(const RestrictedGlobalToLocalDenseMatrix &)                                       = default;
    RestrictedGlobalToLocalDenseMatrix(RestrictedGlobalToLocalDenseMatrix &&RestrictedGlobalToLocalDenseMatrix) noexcept            = default;
    RestrictedGlobalToLocalDenseMatrix &operator=(RestrictedGlobalToLocalDenseMatrix &&RestrictedGlobalToLocalDenseMatrix) noexcept = default;
    ~RestrictedGlobalToLocalDenseMatrix()                                                                                           = default;

    void local_add_vector_product(char trans, CoefficientPrecision alpha, const CoefficientPrecision *in, CoefficientPrecision beta, CoefficientPrecision *out) const override {
        if (m_symmetry == 'N') {
            add_matrix_vector_product(trans, alpha, m_data, in, beta, out);
        } else if (m_symmetry == 'S') {
            add_symmetric_matrix_vector_product(m_UPLO, alpha, m_data, in, beta, out);
        } else if (m_symmetry == 'H') {
            add_hermitian_matrix_vector_product(m_UPLO, alpha, m_data, in, beta, out);
        }
    }

    void local_add_matrix_product_row_major(char trans, CoefficientPrecision alpha, const CoefficientPrecision *in, CoefficientPrecision beta, CoefficientPrecision *out, int mu) const override {
        if (m_symmetry == 'N') {
            add_matrix_matrix_product_row_major(trans, 'N', alpha, m_data, in, beta, out, mu);
        } else if (m_symmetry == 'S') {
            add_symmetric_matrix_matrix_product_row_major('L', m_UPLO, alpha, m_data, in, beta, out, mu);
        } else if (m_symmetry == 'H') {
            add_hermitian_matrix_matrix_product_row_major('L', m_UPLO, alpha, m_data, in, beta, out, mu);
        }
    }
};
} // namespace htool
#endif
