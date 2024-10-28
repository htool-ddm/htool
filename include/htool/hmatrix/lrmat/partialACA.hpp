#ifndef HTOOL_PARTIALACA_HPP
#define HTOOL_PARTIALACA_HPP

#include "../../basic_types/vector.hpp"
#include "../../clustering/cluster_node.hpp"                    // for Cluster
#include "../../hmatrix/interfaces/virtual_generator.hpp"       // for VirtualGenerator
#include "../../hmatrix/interfaces/virtual_lrmat_generator.hpp" // for Virt...
#include "../../matrix/matrix.hpp"                              // for Matrix
#include "../../misc/logger.hpp"                                // for Logger
#include "../../misc/misc.hpp"                                  // for unde...
#include "../../wrappers/wrapper_blas.hpp"                      // for Blas
#include <algorithm>                                            // for min
#include <cmath>                                                // for sqrt
#include <complex>                                              // for real
#include <string>                                               // for oper...
#include <vector>                                               // for vector

namespace htool {

template <typename CoefficientPrecision>
class partialACA final : public VirtualInternalLowRankGenerator<CoefficientPrecision> {

    const VirtualInternalGenerator<CoefficientPrecision> &m_A;

  public:
    using VirtualInternalLowRankGenerator<CoefficientPrecision>::VirtualInternalLowRankGenerator;

    partialACA(const VirtualInternalGenerator<CoefficientPrecision> &A) : m_A(A) {}
    partialACA(const VirtualGenerator<CoefficientPrecision> &A) : m_A(InternalGeneratorWithPermutation<CoefficientPrecision>(A)) {}

    bool copy_low_rank_approximation(int M, int N, int row_offset, int col_offset, LowRankMatrix<CoefficientPrecision> &lrmat) const override {
        int reqrank = -1;
        return copy_low_rank_approximation(M, N, row_offset, col_offset, lrmat.get_epsilon(), reqrank, lrmat);
    }

    bool copy_low_rank_approximation(int M, int N, int row_offset, int col_offset, int reqrank, LowRankMatrix<CoefficientPrecision> &lrmat) const override {
        return copy_low_rank_approximation(M, N, row_offset, col_offset, lrmat.get_epsilon(), reqrank, lrmat);
    }

  private:
    bool copy_low_rank_approximation(int M, int N, int row_offset, int col_offset, underlying_type<CoefficientPrecision> epsilon, int &rank, LowRankMatrix<CoefficientPrecision> &lrmat) const {

        int target_size   = M;
        int source_size   = N;
        int target_offset = row_offset;
        int source_offset = col_offset;

        //// Choice of the first row (see paragraph 3.4.3 page 151 Bebendorf)
        // double dist = 1e30;
        int I = 0;

        // for (int i = 0; i < M; i++) {
        //     double aux_dist = std::sqrt(std::inner_product(xt + (t.get_space_dim() * rows[i]), xt + (t.get_space_dim() * rows[i]) + t.get_space_dim(), t.get_ctr().begin(), double(0), std::plus<double>(), [](double u, double v) { return (u - v) * (u - v); }));
        //     if (dist > aux_dist) {
        //         dist = aux_dist;
        //         I    = i;
        //     }
        // }

        // Partial pivot
        int J       = 0;
        int q       = 0;
        int reqrank = rank;
        std::vector<std::vector<CoefficientPrecision>> uu, vv;
        std::vector<bool> visited_row(target_size, false);
        std::vector<bool> visited_col(source_size, false);

        underlying_type<CoefficientPrecision> frob = 0;
        underlying_type<CoefficientPrecision> aux  = 0;
        underlying_type<CoefficientPrecision> pivot, tmp;
        CoefficientPrecision coef;
        int incx(1), incy(1);
        std::vector<CoefficientPrecision> r(source_size), c(target_size);
        // Either we have a required rank
        // Or it is negative and we have to check the relative error between two iterations.
        // But to do that we need a least two iterations.
        while (((reqrank > 0) && (q < std::min(reqrank, std::min(target_size, source_size)))) || ((reqrank < 0) && (q == 0 || sqrt(aux / frob) > epsilon))) {
            // if (q != 0)
            // std::cout << sqrt(aux / frob) << " " << this->epsilon << " " << (sqrt(aux / frob) > this->epsilon) << std::endl;
            // Next current rank
            q += 1;

            if (q * (target_size + source_size) > (target_size * source_size)) { // the next current rank would not be advantageous
                q = -1;
                break;
            } else {

                // Compute the first cross
                //==================//
                // Look for a column
                std::fill(r.begin(), r.end(), CoefficientPrecision(0));
                m_A.copy_submatrix(1, source_size, I + target_offset, source_offset, r.data());
                for (int j = 0; j < uu.size(); j++) {
                    coef = -uu[j][I];
                    Blas<CoefficientPrecision>::axpy(&(source_size), &(coef), vv[j].data(), &incx, r.data(), &incy);
                }

                pivot = 0.;
                tmp   = 0;
                for (int k = 0; k < source_size; k++) {
                    if (visited_col[k])
                        continue;
                    tmp = std::abs(r[k]);
                    if (tmp < pivot)
                        continue;
                    pivot = tmp;
                    J     = k;
                }

                visited_row[I]             = true;
                CoefficientPrecision gamma = CoefficientPrecision(1.) / r[J];
                //==================//
                // Look for a line
                if (std::abs(r[J]) > 1e-15) {
                    std::fill(c.begin(), c.end(), CoefficientPrecision(0));
                    m_A.copy_submatrix(target_size, 1, target_offset, J + source_offset, c.data());
                    for (int k = 0; k < uu.size(); k++) {
                        coef = -vv[k][J];
                        Blas<CoefficientPrecision>::axpy(&(target_size), &(coef), uu[k].data(), &incx, c.data(), &incy);
                    }
                    c *= gamma;
                    pivot = 0.;
                    tmp   = 0;
                    for (int k = 0; k < target_size; k++) {
                        if (visited_row[k])
                            continue;
                        tmp = std::abs(c[k]);
                        if (tmp < pivot)
                            continue;
                        pivot = tmp;
                        I     = k;
                    }
                    visited_col[J] = true;
                    // Test if no given rank
                    if (reqrank < 0) {
                        // Error estimator
                        CoefficientPrecision frob_aux = 0.;
                        // aux        = std::abs(dprod(c, c) * dprod(r, r));
                        aux = std::abs(Blas<CoefficientPrecision>::dot(&(target_size), c.data(), &incx, c.data(), &incx)) * std::abs(Blas<CoefficientPrecision>::dot(&(source_size), r.data(), &incx, r.data(), &incx));

                        // aux: terme quadratiques du developpement du carre' de la norme de Frobenius de la matrice low rank
                        for (int j = 0; j < uu.size(); j++) {
                            frob_aux += Blas<CoefficientPrecision>::dot(&(source_size), vv[j].data(), &incx, r.data(), &incy) * Blas<CoefficientPrecision>::dot(&(target_size), uu[j].data(), &(incx), c.data(), &(incy));
                        }
                        // frob_aux: termes croises du developpement du carre' de la norme de Frobenius de la matrice low rank
                        frob += aux + 2 * std::real(frob_aux); // frob: Frobenius norm of the low rank matrix
                        // std::cout << frob << " " << aux << " " << frob_aux << std::endl;
                        //==================//
                    }
                    // Matrix<CoefficientPrecision> M=A.get_submatrix(rows,cols);
                    // uu.push_back(M.get_col(J));
                    // vv.push_back(M.get_row(I)/M(I,J));
                    // New cross added
                    uu.push_back(c);
                    vv.push_back(r);

                } else {
                    q -= 1;
                    if (q == 0) { // corner case where first row is zero, ACA fails, we build a dense block instead
                        q = -1;
                    }
                    htool::Logger::get_instance().log(LogLevel::WARNING, "ACA found a zero row in a " + std::to_string(target_size) + "x" + std::to_string(source_size) + " block. Final rank is " + std::to_string(q)); // LCOV_EXCL_LINE
                    break;
                }
            }
        }
        // Final rank
        rank = q;
        if (rank > 0) {
            auto &U = lrmat.get_U();
            auto &V = lrmat.get_V();
            U.resize(target_size, rank);
            V.resize(rank, source_size);
            for (int k = 0; k < rank; k++) {
                std::move(uu[k].begin(), uu[k].end(), U.data() + k * target_size);
                for (int j = 0; j < source_size; j++) {
                    V(k, j) = vv[k][j];
                }
            }
            return true;
        }
        return false;
    }
};
} // namespace htool
#endif
