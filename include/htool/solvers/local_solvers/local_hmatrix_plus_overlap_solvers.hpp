#ifndef HTOOL_SOLVERS_LOCAL_SOLVERS_HMATRIX_PLUS_OVERLAP_HPP
#define HTOOL_SOLVERS_LOCAL_SOLVERS_HMATRIX_PLUS_OVERLAP_HPP

#include "../interfaces/virtual_local_solver.hpp"
#include "htool/hmatrix/hmatrix.hpp"
#include "htool/hmatrix/linalg/triangular_hmatrix_matrix_solve.hpp"
#include "htool/matrix/linalg/factorization.hpp"
#include "htool/matrix/matrix.hpp"
#include "htool/misc/misc.hpp"
#include <algorithm>

namespace htool {

template <typename CoefficientPrecision, typename CoordinatePrecision = underlying_type<CoefficientPrecision>>
class LocalHMatrixPlusOverlapSolver : public VirtualLocalSolver<CoefficientPrecision> {
  private:
    HMatrix<CoefficientPrecision, CoordinatePrecision> &m_local_hmatrix;
    Matrix<CoefficientPrecision> &m_B, &m_C, &m_D;
    mutable Matrix<CoefficientPrecision> buffer;

  public:
    LocalHMatrixPlusOverlapSolver(HMatrix<CoefficientPrecision> &local_hmatrix, Matrix<CoefficientPrecision> &B, Matrix<CoefficientPrecision> &C, Matrix<CoefficientPrecision> &D) : m_local_hmatrix(local_hmatrix), m_B(B), m_C(C), m_D(D) {}
    void numfact(HPDDM::MatrixCSR<CoefficientPrecision> *const &, bool = false, CoefficientPrecision *const & = nullptr) {
        if (m_local_hmatrix.get_symmetry() == 'N') {
            lu_factorization(m_local_hmatrix);
            if (m_C.nb_rows() > 0) {
                internal_triangular_hmatrix_matrix_solve('L', 'L', 'N', 'U', CoefficientPrecision(1), m_local_hmatrix, m_B);
                internal_triangular_hmatrix_matrix_solve('R', 'U', 'N', 'N', CoefficientPrecision(1), m_local_hmatrix, m_C);
                add_matrix_matrix_product('N', 'N', CoefficientPrecision(-1), m_C, m_B, CoefficientPrecision(1), m_D);
                lu_factorization(m_D);
            }

        } else if (m_local_hmatrix.get_symmetry() == 'S' || m_local_hmatrix.get_symmetry() == 'H') {
            cholesky_factorization(m_local_hmatrix.get_UPLO(), m_local_hmatrix);
            if (m_local_hmatrix.get_UPLO() == 'L' && m_C.nb_rows() > 0) {
                internal_triangular_hmatrix_matrix_solve('R', m_local_hmatrix.get_UPLO(), is_complex<CoefficientPrecision>() ? 'C' : 'T', 'N', CoefficientPrecision(1), m_local_hmatrix, m_C);
                add_matrix_matrix_product('N', is_complex<CoefficientPrecision>() ? 'C' : 'T', CoefficientPrecision(-1), m_C, m_C, CoefficientPrecision(1), m_D);
                cholesky_factorization(m_local_hmatrix.get_UPLO(), m_D);
            } else if (m_local_hmatrix.get_UPLO() == 'U' && m_B.nb_cols() > 0) {
                internal_triangular_hmatrix_matrix_solve('L', m_local_hmatrix.get_UPLO(), is_complex<CoefficientPrecision>() ? 'C' : 'T', 'N', CoefficientPrecision(1), m_local_hmatrix, m_B);
                add_matrix_matrix_product(is_complex<CoefficientPrecision>() ? 'C' : 'T', 'N', CoefficientPrecision(-1), m_B, m_B, CoefficientPrecision(1), m_D);
                cholesky_factorization(m_local_hmatrix.get_UPLO(), m_D);
            }
        }
    }
    void solve(CoefficientPrecision *const b, const unsigned short &mu = 1) const {
        int local_size_wo_overlap = m_local_hmatrix.get_target_cluster().get_size();
        int size_overlap          = m_D.nb_rows();
        int local_size_w_overlap  = local_size_wo_overlap + size_overlap;
        Matrix<CoefficientPrecision> b1(local_size_wo_overlap, mu), b2(size_overlap, mu);

        for (int i = 0; i < mu; i++) {
            std::copy_n(b + i * local_size_w_overlap, local_size_wo_overlap, b1.data() + i * local_size_wo_overlap);
            std::copy_n(b + i * local_size_w_overlap + local_size_wo_overlap, size_overlap, b2.data() + i * size_overlap);
        }

        this->solve(b1, b2);

        for (int i = 0; i < mu; i++) {
            std::copy_n(b1.data() + i * local_size_wo_overlap, local_size_wo_overlap, b + i * local_size_w_overlap);
            std::copy_n(b2.data() + i * size_overlap, size_overlap, b + i * local_size_w_overlap + local_size_wo_overlap);
        }
    }
    void solve(const CoefficientPrecision *const b, CoefficientPrecision *const x, const unsigned short &mu = 1) const {

        int local_size_wo_overlap = m_local_hmatrix.get_target_cluster().get_size();
        int size_overlap          = m_D.nb_rows();
        int local_size_w_overlap  = local_size_wo_overlap + size_overlap;
        Matrix<CoefficientPrecision> b1(local_size_wo_overlap, mu), b2(size_overlap, mu);

        for (int i = 0; i < mu; i++) {
            std::copy_n(b + i * local_size_w_overlap, local_size_wo_overlap, b1.data() + i * local_size_wo_overlap);
            std::copy_n(b + i * local_size_w_overlap + local_size_wo_overlap, size_overlap, b2.data() + i * size_overlap);
        }
        this->solve(b1, b2);

        for (int i = 0; i < mu; i++) {
            std::copy_n(b1.data() + i * local_size_wo_overlap, local_size_wo_overlap, x + i * local_size_w_overlap);
            std::copy_n(b2.data() + i * size_overlap, size_overlap, x + i * local_size_w_overlap + local_size_wo_overlap);
        }
    }

  private:
    void solve(Matrix<CoefficientPrecision> &b1, Matrix<CoefficientPrecision> &b2) const {

        if (m_local_hmatrix.get_symmetry() == 'N') {

            internal_triangular_hmatrix_matrix_solve('L', 'L', 'N', 'U', CoefficientPrecision(1), m_local_hmatrix, b1);
            if (m_C.nb_rows() > 0) {
                add_matrix_matrix_product('N', 'N', CoefficientPrecision(-1), m_C, b1, CoefficientPrecision(1), b2);
                triangular_matrix_matrix_solve('L', 'L', 'N', 'U', CoefficientPrecision(1), m_D, b2);

                triangular_matrix_matrix_solve('L', 'U', 'N', 'N', CoefficientPrecision(1), m_D, b2);
                add_matrix_matrix_product('N', 'N', CoefficientPrecision(-1), m_B, b2, CoefficientPrecision(1), b1);
            }
            internal_triangular_hmatrix_matrix_solve('L', 'U', 'N', 'N', CoefficientPrecision(1), m_local_hmatrix, b1);
        } else if ((m_local_hmatrix.get_symmetry() == 'S' || m_local_hmatrix.get_symmetry() == 'H') && m_local_hmatrix.get_UPLO() == 'L') {
            internal_triangular_hmatrix_matrix_solve('L', 'L', 'N', 'N', CoefficientPrecision(1), m_local_hmatrix, b1);

            if (m_C.nb_rows() > 0) {
                add_matrix_matrix_product('N', 'N', CoefficientPrecision(-1), m_C, b1, CoefficientPrecision(1), b2);
                triangular_matrix_matrix_solve('L', 'L', 'N', 'U', CoefficientPrecision(1), m_D, b2);
                triangular_matrix_matrix_solve('L', 'L', is_complex<CoefficientPrecision>() ? 'C' : 'T', 'N', CoefficientPrecision(1), m_D, b2);
                add_matrix_matrix_product(is_complex<CoefficientPrecision>() ? 'C' : 'T', 'N', CoefficientPrecision(-1), m_C, b2, CoefficientPrecision(1), b1);
            }
            internal_triangular_hmatrix_matrix_solve('L', 'L', is_complex<CoefficientPrecision>() ? 'C' : 'T', 'N', CoefficientPrecision(1), m_local_hmatrix, b1);
        } else if ((m_local_hmatrix.get_symmetry() == 'S' || m_local_hmatrix.get_symmetry() == 'H') && m_local_hmatrix.get_UPLO() == 'U') {
            internal_triangular_hmatrix_matrix_solve('L', 'U', is_complex<CoefficientPrecision>() ? 'C' : 'T', 'N', CoefficientPrecision(1), m_local_hmatrix, b1);

            if (m_B.nb_cols() > 0) {
                add_matrix_matrix_product(is_complex<CoefficientPrecision>() ? 'C' : 'T', 'N', CoefficientPrecision(-1), m_B, b1, CoefficientPrecision(1), b2);
                triangular_matrix_matrix_solve('L', 'U', 'N', 'U', CoefficientPrecision(1), m_D, b2);
                add_matrix_matrix_product('N', 'N', CoefficientPrecision(-1), m_B, b2, CoefficientPrecision(1), b1);
            }

            internal_triangular_hmatrix_matrix_solve('L', 'U', 'N', 'N', CoefficientPrecision(1), m_local_hmatrix, b1);
        }
    }
};
} // namespace htool
#endif
