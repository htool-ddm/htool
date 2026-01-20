#ifndef HTOOL_VIRTUAL_IBCOMP_HPP
#define HTOOL_VIRTUAL_IBCOMP_HPP

#include "../../hmatrix/interfaces/virtual_lrmat_generator.hpp" // for Virt...
#include "../../wrappers/wrapper_blas.hpp"                      // for Blas
#include "../../wrappers/wrapper_lapack.hpp"                    // for Blas
#include "lrmat.hpp"
#include <map>
#include <span>

namespace htool {

inline double cheb_node(int k, int L) { return std::cos((double)(2 * k + 1) / (double)(2 * L) * M_PI); }
inline double T(double x, int k) { return cos((double)(k)*acos(x)); }
inline void getTr(int L, double *Tr) {
    for (int k = 0; k < L; k++) {
        double r = cheb_node(k, L);
        for (int i = 0; i < L; i++) {
            (Tr + k * L)[i] = T(r, i);
        }
    }
}
inline void getTs(double u, int L, double *Tu) {
    for (int i = 0; i < L; i++) {
        Tu[i] = T(u, i);
    }
}
inline void C1D(double *Tx, double *Tr, int L, double *S) {
    for (int k = 0; k < L; k++) {
        double res  = 1.;
        double *Trk = Tr + k * L;
        for (int j = 1; j < L; j++) {
            res += 2. * Tx[j] * Trk[j];
        }
        S[k] = res / (double)(L);
    }
}
inline void part_to_cheb(double x, double y, double z, double *Mu, int Lc) {
    double *Cx = new double[Lc];
    double *Cy = new double[Lc];
    double *Cz = new double[Lc];
    for (int i = 0; i < Lc; i++) {
        Cx[i] = C1D(i, x, Lc);
        Cy[i] = C1D(i, y, Lc);
        Cz[i] = C1D(i, z, Lc);
    }
    for (int i = 0; i < Lc; i++) {
        for (int j = 0; j < Lc; j++) {
            for (int k = 0; k < Lc; k++) {
                Mu[i * Lc * Lc + j * Lc + k] = Cx[i] * Cy[j] * Cz[k];
            }
        }
    }
}

template <typename CoefficientPrecision, typename CoordinatePrecision, int dimension>
class IBcomp final : public VirtualInternalLowRankGenerator<CoefficientPrecision> {

  private:
    const int *m_target_permutation;
    const int *m_source_permutation;
    std::function<CoefficientPrecision(CoordinatePrecision *, CoordinatePrecision *)> m_kernel;
    const std::map<int, std::vector<std::pair<int, CoordinatePrecision>>> &
        m_target_dof_to_elts;
    const std::map<int, std::vector<std::pair<int, CoordinatePrecision>>> &m_source_dof_to_elts;

  public:
    IBcomp(std::function<CoefficientPrecision(CoordinatePrecision *, CoordinatePrecision *)> kernel, const int *target_permutation, const std::map<int, std::vector<std::pair<int, CoordinatePrecision>>> &target_dof_to_elts, const int *source_permutation, const std::map<int, std::vector<std::pair<int, CoordinatePrecision>>> &source_dof_to_elts) : m_kernel(kernel), m_target_permutation(target_permutation), m_target_dof_to_elts(target_dof_to_elts), m_source_permutation(source_permutation), m_source_dof_to_elts(source_dof_to_elts) {}

    virtual bool copy_low_rank_approximation(int M, int N, int row_offset, int col_offset, LowRankMatrix<CoefficientPrecision> &lrmat) const override {
        return copy_low_rank_approximation(M, N, m_target_permutation + row_offset, m_source_permutation + col_offset, lrmat);
    }

    virtual bool copy_low_rank_approximation(int M, int N, int row_offset, int col_offset, int reqrank, LowRankMatrix<CoefficientPrecision> &lrmat) const override {
        return copy_low_rank_approximation(M, N, m_target_permutation + row_offset, m_source_permutation + col_offset, reqrank, lrmat);
    }

    // C style
    bool copy_low_rank_approximation(int M, int N, const int *rows, const int *cols, LowRankMatrix<CoefficientPrecision> &lrmat) const {
        std::array<CoordinatePrecision, 2 * dimension> target_box;
        std::array<CoordinatePrecision, 2 * dimension> source_box;
        for (int dim = 0; dim < dimension; dim++) {
            target_box[2 * dim + 0] = std::numeric_limits<double>::min();
            target_box[2 * dim + 1] = std::numeric_limits<double>::max();
            source_box[2 * dim + 0] = std::numeric_limits<double>::min();
            source_box[2 * dim + 1] = std::numeric_limits<double>::max();
        }
        for (int target_index : std::span<int>(rows, N)) {
            auto &target_elts = m_target_dof_to_elts[target_index];
            for (auto [_, elt] : target_elts) {
                for (int dim = 0; dim < dimension; dim++) {
                    target_box[2 * dim + 0] = std::max(target_box[2 * dim + 0], elt[dim]);
                    target_box[2 * dim + 1] = std::min(target_box[2 * dim + 1], elt[dim]);
                }
            }
        }
        for (int source_index : std::span<int>(cols, N)) {
            auto &source_elts = m_source_dof_to_elts[source_index];
            for (auto [_, elt] : source_elts) {
                for (int dim = 0; dim < dimension; dim++) {
                    source_box[2 * dim + 0] = std::max(source_box[2 * dim + 0], elt[dim]);
                    source_box[2 * dim + 1] = std::min(source_box[2 * dim + 1], elt[dim]);
                }
            }
        }
    }

    bool copy_low_rank_approximation(int M, int N, const int *rows, const int *cols, int reqrank, LowRankMatrix<CoefficientPrecision> &lrmat) const {}
};

} // namespace htool

#endif
