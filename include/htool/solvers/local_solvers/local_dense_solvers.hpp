#ifndef HTOOL_SOLVERS_LOCAL_SOLVER_DENSE_HPP
#define HTOOL_SOLVERS_LOCAL_SOLVER_DENSE_HPP

#include "../../matrix/linalg/factorization.hpp"
#include "../interfaces/virtual_local_solver.hpp"

namespace htool {

template <typename CoefficientPrecision>
class LocalDenseSolver : public VirtualLocalSolver<CoefficientPrecision> {
  private:
    Matrix<CoefficientPrecision> &m_local_matrix;

  public:
    LocalDenseSolver(Matrix<CoefficientPrecision> &local_matrix) : m_local_matrix(local_matrix) {}
    void numfact(HPDDM::MatrixCSR<CoefficientPrecision> *const &, bool = false, CoefficientPrecision *const & = nullptr) {
        lu_factorization(m_local_matrix);
    }
    void solve(CoefficientPrecision *const b, const unsigned short &mu = 1) const {
        Matrix<CoefficientPrecision> b_view;
        b_view.assign(m_local_matrix.nb_cols(), mu, b, false);
        lu_solve('N', m_local_matrix, b_view);
    }
    void solve(const CoefficientPrecision *const b, CoefficientPrecision *const x, const unsigned short &mu = 1) const {
        std::copy_n(b, m_local_matrix.nb_cols() * mu, x);
        Matrix<CoefficientPrecision> b_view;
        b_view.assign(m_local_matrix.nb_cols(), mu, x, false);
        lu_solve('N', m_local_matrix, b_view);
    }
};
} // namespace htool
#endif
