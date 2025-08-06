#ifndef HTOOL_SOLVERS_LOCAL_SOLVERS_HMATRIX_HPP
#define HTOOL_SOLVERS_LOCAL_SOLVERS_HMATRIX_HPP

#include "../interfaces/virtual_local_solver.hpp" // for VirtualLocalSolver
#include "htool/clustering/cluster_node.hpp"      // for cluster_to_user
#include "htool/hmatrix/hmatrix.hpp"              // for HMatrix
#include "htool/hmatrix/linalg/factorization.hpp" // for lu_factorization
#include "htool/matrix/matrix.hpp"                // for Matrix
#include "htool/misc/misc.hpp"                    // for underlying_type
#include <algorithm>                              // for copy_n

namespace htool {

template <typename CoefficientPrecision, typename CoordinatePrecision = underlying_type<CoefficientPrecision>>
class LocalHMatrixSolver : public VirtualLocalSolver<CoefficientPrecision> {
  private:
    HMatrix<CoefficientPrecision, CoordinatePrecision> &m_local_hmatrix;
    bool m_is_using_permutation;
    mutable Matrix<CoefficientPrecision> buffer;

  public:
    LocalHMatrixSolver(HMatrix<CoefficientPrecision> &local_hmatrix, bool is_using_permutation) : m_local_hmatrix(local_hmatrix), m_is_using_permutation(is_using_permutation) {}
    void numfact(HPDDM::MatrixCSR<CoefficientPrecision> *const &, bool = false, CoefficientPrecision *const & = nullptr) {
        if (m_local_hmatrix.get_symmetry() == 'N') {
            sequential_lu_factorization(m_local_hmatrix);
        } else if (m_local_hmatrix.get_symmetry() == 'S' || m_local_hmatrix.get_symmetry() == 'H') {
            sequential_cholesky_factorization(m_local_hmatrix.get_UPLO(), m_local_hmatrix);
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
        } else if (m_local_hmatrix.get_symmetry() == 'S' || m_local_hmatrix.get_symmetry() == 'H') {
            internal_cholesky_solve(m_local_hmatrix.get_UPLO(), m_local_hmatrix, buffer);
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
        } else if (m_local_hmatrix.get_symmetry() == 'S' || m_local_hmatrix.get_symmetry() == 'H') {
            internal_cholesky_solve(m_local_hmatrix.get_UPLO(), m_local_hmatrix, buffer);
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
