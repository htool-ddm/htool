#ifndef HTOOL_SOLVERS_LOCAL_SOLVERS_HMATRIX_HPP
#define HTOOL_SOLVERS_LOCAL_SOLVERS_HMATRIX_HPP

#include "../interfaces/virtual_local_solver.hpp" // for VirtualLocalSolver
#include "htool/clustering/cluster_node.hpp"      // for cluster_to_user
#include "htool/hmatrix/hmatrix.hpp"              // for HMatrix
#include "htool/hmatrix/linalg/factorization.hpp" // for lu_factorization
#include "htool/matrix/matrix.hpp"                // for Matrix
#include "htool/misc/logger.hpp"                  // for Logger
#include "htool/misc/misc.hpp"                    // for underlying_type
#include <algorithm>                              // for copy_n

namespace htool {

// Whether a symmetric/Hermitian local HMatrix is factorized with Cholesky rather than LDLt: only when
// declared positive definite. A complex symmetric (non-Hermitian) matrix cannot be positive definite.
template <typename CoefficientPrecision, typename CoordinatePrecision>
bool use_cholesky_for_local_hmatrix(const HMatrix<CoefficientPrecision, CoordinatePrecision> &local_hmatrix, bool is_positive_definite) {
    char symmetry = local_hmatrix.get_symmetry();
    if (is_positive_definite and symmetry == 'S' and is_complex<CoefficientPrecision>()) {
        htool::Logger::get_instance().log(LogLevel::WARNING, "is_positive_definite is ignored for a complex symmetric (non-Hermitian) local matrix, which is factorized with LDLt."); // LCOV_EXCL_LINE
    }
    return is_positive_definite and (symmetry == 'H' or (symmetry == 'S' and !is_complex<CoefficientPrecision>()));
}

template <typename CoefficientPrecision, typename CoordinatePrecision = underlying_type<CoefficientPrecision>>
class LocalHMatrixSolver : public VirtualLocalSolver<CoefficientPrecision> {
  private:
    HMatrix<CoefficientPrecision, CoordinatePrecision> &m_local_hmatrix;
    bool m_is_using_permutation;
    bool m_is_positive_definite;
    bool m_use_cholesky{false};
    mutable Matrix<CoefficientPrecision> buffer;

  public:
    // is_positive_definite: a symmetric/Hermitian local_hmatrix is factorized with Cholesky instead of LDLt.
    LocalHMatrixSolver(HMatrix<CoefficientPrecision> &local_hmatrix, bool is_using_permutation, bool is_positive_definite = false) : m_local_hmatrix(local_hmatrix), m_is_using_permutation(is_using_permutation), m_is_positive_definite(is_positive_definite) {}
    void numfact(HPDDM::MatrixCSR<CoefficientPrecision> *const &, bool = false, CoefficientPrecision *const & = nullptr) {
        m_use_cholesky = use_cholesky_for_local_hmatrix(m_local_hmatrix, m_is_positive_definite);
        if (m_local_hmatrix.get_symmetry() == 'N') {
            lu_factorization(m_local_hmatrix);
        } else if (m_use_cholesky) {
            cholesky_factorization(m_local_hmatrix.get_UPLO(), m_local_hmatrix);
        } else {
            ldlt_factorization(m_local_hmatrix.get_symmetry(), m_local_hmatrix.get_UPLO(), m_local_hmatrix);
        }
    }
    void solve(CoefficientPrecision *const b, const unsigned short &mu = 1) const {

        if (m_is_using_permutation) {
            if (buffer.nb_rows() != m_local_hmatrix.nb_cols() or buffer.nb_cols() != mu) {
                buffer.resize(m_local_hmatrix.nb_cols(), mu);
            }

            auto &source_cluster = m_local_hmatrix.get_source_cluster();
            for (int i = 0; i < mu; i++) {
                user_to_cluster(source_cluster, b + source_cluster.get_size() * i, buffer.data() + source_cluster.get_size() * i);
            }
        } else {
            buffer.assign(m_local_hmatrix.nb_cols(), mu, b, false);
        }

        if (m_local_hmatrix.get_symmetry() == 'N') {
            internal_lu_solve('N', m_local_hmatrix, buffer);
        } else if (m_use_cholesky) {
            internal_cholesky_solve(m_local_hmatrix.get_UPLO(), m_local_hmatrix, buffer);
        } else {
            internal_ldlt_solve(m_local_hmatrix.get_symmetry(), m_local_hmatrix.get_UPLO(), m_local_hmatrix, buffer);
        }

        if (m_is_using_permutation) {
            auto &target_cluster = m_local_hmatrix.get_target_cluster();
            for (int i = 0; i < mu; i++) {
                cluster_to_user(target_cluster, buffer.data() + target_cluster.get_size() * i, b + target_cluster.get_size() * i);
            }
        }
    }
    void solve(const CoefficientPrecision *const b, CoefficientPrecision *const x, const unsigned short &mu = 1) const {
        if (buffer.nb_rows() != m_local_hmatrix.nb_cols() or buffer.nb_cols() != mu) {
            buffer.resize(m_local_hmatrix.nb_cols(), mu);
        }
        if (m_is_using_permutation) {
            auto &source_cluster = m_local_hmatrix.get_source_cluster();
            for (int i = 0; i < mu; i++) {
                user_to_cluster(source_cluster, b + source_cluster.get_size() * i, buffer.data() + source_cluster.get_size() * i);
            }
        } else {
            buffer.assign(m_local_hmatrix.nb_cols(), mu, x, false);
            std::copy_n(b, mu * m_local_hmatrix.nb_cols(), buffer.data());
        }

        if (m_local_hmatrix.get_symmetry() == 'N') {
            internal_lu_solve('N', m_local_hmatrix, buffer);
        } else if (m_use_cholesky) {
            internal_cholesky_solve(m_local_hmatrix.get_UPLO(), m_local_hmatrix, buffer);
        } else {
            internal_ldlt_solve(m_local_hmatrix.get_symmetry(), m_local_hmatrix.get_UPLO(), m_local_hmatrix, buffer);
        }

        if (m_is_using_permutation) {
            auto &target_cluster = m_local_hmatrix.get_target_cluster();
            for (int i = 0; i < mu; i++) {
                cluster_to_user(target_cluster, buffer.data() + target_cluster.get_size() * i, x + target_cluster.get_size() * i);
            }
        }
    }
};
} // namespace htool
#endif
