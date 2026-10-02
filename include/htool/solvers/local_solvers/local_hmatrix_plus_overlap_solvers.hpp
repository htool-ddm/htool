#ifndef HTOOL_SOLVERS_LOCAL_SOLVERS_HMATRIX_PLUS_OVERLAP_HPP
#define HTOOL_SOLVERS_LOCAL_SOLVERS_HMATRIX_PLUS_OVERLAP_HPP

#include "../interfaces/virtual_local_solver.hpp"
#include "htool/hmatrix/hmatrix.hpp"
#include "htool/hmatrix/linalg/factorization.hpp"
#include "htool/hmatrix/linalg/triangular_hmatrix_matrix_solve.hpp"
#include "htool/matrix/linalg/factorization.hpp"
#include "htool/matrix/matrix.hpp"
#include "htool/misc/misc.hpp"
#include "local_hmatrix_solvers.hpp"
#include <algorithm>

namespace htool {

template <typename CoefficientPrecision, typename CoordinatePrecision = underlying_type<CoefficientPrecision>>
class LocalHMatrixPlusOverlapSolver : public VirtualLocalSolver<CoefficientPrecision> {
  private:
    HMatrix<CoefficientPrecision, CoordinatePrecision> &m_local_hmatrix;
    Matrix<CoefficientPrecision> &m_B, &m_C, &m_D;
    bool m_is_positive_definite;
    bool m_use_cholesky{false};
    mutable Matrix<CoefficientPrecision> buffer;

  public:
    // is_positive_definite: a symmetric/Hermitian local_hmatrix (and its overlap Schur complement) is
    // factorized with Cholesky instead of LDLt.
    LocalHMatrixPlusOverlapSolver(HMatrix<CoefficientPrecision> &local_hmatrix, Matrix<CoefficientPrecision> &B, Matrix<CoefficientPrecision> &C, Matrix<CoefficientPrecision> &D, bool is_positive_definite = false) : m_local_hmatrix(local_hmatrix), m_B(B), m_C(C), m_D(D), m_is_positive_definite(is_positive_definite) {}
    void numfact(HPDDM::MatrixCSR<CoefficientPrecision> *const &, bool = false, CoefficientPrecision *const & = nullptr) {
        m_use_cholesky = use_cholesky_for_local_hmatrix(m_local_hmatrix, m_is_positive_definite);
        if (m_local_hmatrix.get_symmetry() == 'N') {
            lu_factorization(m_local_hmatrix);
            if (m_C.nb_rows() > 0) {
                internal_triangular_hmatrix_matrix_solve('L', 'L', 'N', 'U', CoefficientPrecision(1), m_local_hmatrix, m_B);
                internal_triangular_hmatrix_matrix_solve('R', 'U', 'N', 'N', CoefficientPrecision(1), m_local_hmatrix, m_C);
                add_matrix_matrix_product('N', 'N', CoefficientPrecision(-1), m_C, m_B, CoefficientPrecision(1), m_D);
                lu_factorization(m_D);
            }

        } else if (!m_use_cholesky) {
            // Block LDLt of [A, C^T; C, D] (C^H if Hermitian) with A = F D_A F^T: C := C F^-T D_A^-1 (or
            // B := D_A^-1 F^-1 B for UPLO='U'), then D -= C (C F^-T)^T, whose LDLt is computed densely.
            char symmetry = m_local_hmatrix.get_symmetry();
            char UPLO     = m_local_hmatrix.get_UPLO();
            char transa   = symmetry == 'H' ? 'C' : 'T';
            int offset    = m_local_hmatrix.get_target_cluster().get_offset();
            ldlt_factorization(symmetry, UPLO, m_local_hmatrix);
            if (UPLO == 'L' && m_C.nb_rows() > 0) {
                internal_triangular_ldlt_hmatrix_matrix_solve('R', UPLO, transa, CoefficientPrecision(1), m_local_hmatrix, m_C);
                Matrix<CoefficientPrecision> W(m_C);
                internal_apply_ldlt_diagonal(symmetry, 'R', UPLO, m_local_hmatrix, m_C, offset);
                add_matrix_matrix_product('N', transa, CoefficientPrecision(-1), m_C, W, CoefficientPrecision(1), m_D);
                ldlt_factorization(symmetry, UPLO, m_D);
            } else if (UPLO == 'U' && m_B.nb_cols() > 0) {
                internal_triangular_ldlt_hmatrix_matrix_solve('L', UPLO, 'N', CoefficientPrecision(1), m_local_hmatrix, m_B);
                Matrix<CoefficientPrecision> V(m_B);
                internal_apply_ldlt_diagonal(symmetry, 'L', UPLO, m_local_hmatrix, m_B, offset);
                add_matrix_matrix_product(transa, 'N', CoefficientPrecision(-1), m_B, V, CoefficientPrecision(1), m_D);
                ldlt_factorization(symmetry, UPLO, m_D);
            }
        } else {
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
        } else if (!m_use_cholesky) {
            char symmetry     = m_local_hmatrix.get_symmetry();
            char UPLO         = m_local_hmatrix.get_UPLO();
            char transa       = symmetry == 'H' ? 'C' : 'T';
            int offset        = m_local_hmatrix.get_target_cluster().get_offset();
            bool with_overlap = UPLO == 'L' ? m_C.nb_rows() > 0 : m_B.nb_cols() > 0;
            internal_triangular_ldlt_hmatrix_matrix_solve('L', UPLO, 'N', CoefficientPrecision(1), m_local_hmatrix, b1);
            if (with_overlap) {
                add_matrix_matrix_product(UPLO == 'L' ? 'N' : transa, 'N', CoefficientPrecision(-1), UPLO == 'L' ? m_C : m_B, b1, CoefficientPrecision(1), b2);
            }
            internal_apply_ldlt_diagonal(symmetry, 'L', UPLO, m_local_hmatrix, b1, offset);
            if (with_overlap) {
                ldlt_solve(symmetry, UPLO, m_D, b2);
                add_matrix_matrix_product(UPLO == 'L' ? transa : 'N', 'N', CoefficientPrecision(-1), UPLO == 'L' ? m_C : m_B, b2, CoefficientPrecision(1), b1);
            }
            internal_triangular_ldlt_hmatrix_matrix_solve('L', UPLO, transa, CoefficientPrecision(1), m_local_hmatrix, b1);
        } else if (m_local_hmatrix.get_UPLO() == 'L') {
            internal_triangular_hmatrix_matrix_solve('L', 'L', 'N', 'N', CoefficientPrecision(1), m_local_hmatrix, b1);

            if (m_C.nb_rows() > 0) {
                add_matrix_matrix_product('N', 'N', CoefficientPrecision(-1), m_C, b1, CoefficientPrecision(1), b2);
                triangular_matrix_matrix_solve('L', 'L', 'N', 'N', CoefficientPrecision(1), m_D, b2);
                triangular_matrix_matrix_solve('L', 'L', is_complex<CoefficientPrecision>() ? 'C' : 'T', 'N', CoefficientPrecision(1), m_D, b2);
                add_matrix_matrix_product(is_complex<CoefficientPrecision>() ? 'C' : 'T', 'N', CoefficientPrecision(-1), m_C, b2, CoefficientPrecision(1), b1);
            }
            internal_triangular_hmatrix_matrix_solve('L', 'L', is_complex<CoefficientPrecision>() ? 'C' : 'T', 'N', CoefficientPrecision(1), m_local_hmatrix, b1);
        } else if (m_local_hmatrix.get_UPLO() == 'U') {
            internal_triangular_hmatrix_matrix_solve('L', 'U', is_complex<CoefficientPrecision>() ? 'C' : 'T', 'N', CoefficientPrecision(1), m_local_hmatrix, b1);

            if (m_B.nb_cols() > 0) {
                add_matrix_matrix_product(is_complex<CoefficientPrecision>() ? 'C' : 'T', 'N', CoefficientPrecision(-1), m_B, b1, CoefficientPrecision(1), b2);
                triangular_matrix_matrix_solve('L', 'U', is_complex<CoefficientPrecision>() ? 'C' : 'T', 'N', CoefficientPrecision(1), m_D, b2);
                triangular_matrix_matrix_solve('L', 'U', 'N', 'N', CoefficientPrecision(1), m_D, b2);
                add_matrix_matrix_product('N', 'N', CoefficientPrecision(-1), m_B, b2, CoefficientPrecision(1), b1);
            }

            internal_triangular_hmatrix_matrix_solve('L', 'U', 'N', 'N', CoefficientPrecision(1), m_local_hmatrix, b1);
        }
    }
};
} // namespace htool
#endif
