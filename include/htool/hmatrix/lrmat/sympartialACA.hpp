#ifndef HTOOL_SYMPARTIALACA_HPP
#define HTOOL_SYMPARTIALACA_HPP

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
#include <string>                                               // for to_s...
#include <vector>                                               // for vector

namespace htool {

template <typename CoefficientPrecision>
class sympartialACA final : public VirtualInternalLowRankGenerator<CoefficientPrecision> {
    std::unique_ptr<InternalGeneratorWithPermutation<CoefficientPrecision>> internal_generator_w_permutation;
    const VirtualInternalGenerator<CoefficientPrecision> &m_A;

  public:
    using VirtualInternalLowRankGenerator<CoefficientPrecision>::VirtualInternalLowRankGenerator;

    sympartialACA(const VirtualInternalGenerator<CoefficientPrecision> &A) : m_A(A) {}
    sympartialACA(const VirtualGenerator<CoefficientPrecision> &A, const int *target_permutation, const int *source_permutation) : internal_generator_w_permutation(std::make_unique<InternalGeneratorWithPermutation<CoefficientPrecision>>(A, target_permutation, source_permutation)), m_A(*internal_generator_w_permutation) {}

    bool copy_low_rank_approximation(int M, int N, int row_offset, int col_offset, LowRankMatrix<CoefficientPrecision> &lrmat) const override {
        int reqrank = -1;
        return copy_low_rank_approximation(M, N, row_offset, col_offset, lrmat.get_epsilon(), reqrank, lrmat);
    }

    bool copy_low_rank_approximation(int M, int N, int row_offset, int col_offset, int reqrank, LowRankMatrix<CoefficientPrecision> &lrmat) const override {
        return copy_low_rank_approximation(M, N, row_offset, col_offset, lrmat.get_epsilon(), reqrank, lrmat);
    }

  private:
    bool copy_low_rank_approximation(int M, int N, int row_offset, int col_offset, underlying_type<CoefficientPrecision> epsilon, int &rank, LowRankMatrix<CoefficientPrecision> &lrmat) const {

        int n1, n2;
        int i1;
        int i2;
        // const double *x1;

        if (row_offset >= col_offset) {

            n1 = M;
            n2 = N;
            i1 = row_offset;
            i2 = col_offset;
            // x1        = xt;
            // cluster_1 = &t;
        } else {
            n1 = N;
            n2 = M;
            i1 = col_offset;
            i2 = row_offset;
            // x1        = xs;
            // cluster_1 = &s;
        }

        //// Choice of the first row (see paragraph 3.4.3 page 151 Bebendorf)
        // double dist = 1e30;
        int I1 = 0;
        // for (int i = 0; i < n1; i++) {
        //     double aux_dist = std::sqrt(std::inner_product(x1 + (cluster_1->get_space_dim() * i1[i]), x1 + (cluster_1->get_space_dim() * i1[i]) + cluster_1->get_space_dim(), cluster_1->get_ctr().begin(), double(0), std::plus<double>(), [](double u, double v) { return (u - v) * (u - v); }));

        //     if (dist > aux_dist) {
        //         dist = aux_dist;
        //         I1   = i;
        //     }
        // }
        // Partial pivot
        int I2      = 0;
        int q       = 0;
        int reqrank = rank;
        std::vector<std::vector<CoefficientPrecision>> uu, vv;
        std::vector<bool> visited_1(n1, false);
        std::vector<bool> visited_2(n2, false);

        underlying_type<CoefficientPrecision> frob = 0;
        underlying_type<CoefficientPrecision> aux  = 0;

        underlying_type<CoefficientPrecision> pivot, tmp;
        CoefficientPrecision coef;
        int incx(1), incy(1);
        std::vector<CoefficientPrecision> u1(n2), u2(n1);

        // Either we have a required rank
        // Or it is negative and we have to check the relative error between two iterations.
        // But to do that we need a least two iterations.
        while (((reqrank > 0) && (q < std::min(reqrank, std::min(n1, n2)))) || ((reqrank < 0) && (q == 0 || sqrt(aux / frob) > epsilon))) {

            // Next current rank
            q += 1;

            if (q * (n1 + n2) > (n1 * n2)) { // the next current rank would not be advantageous
                q = -1;
                break;
            } else {
                std::fill(u1.begin(), u1.end(), CoefficientPrecision(0));
                if (row_offset >= col_offset) {
                    m_A.copy_submatrix(1, n2, i1 + I1, i2, u1.data());
                } else {
                    m_A.copy_submatrix(n2, 1, i2, i1 + I1, u1.data());
                }

                for (int j = 0; j < uu.size(); j++) {
                    coef = -uu[j][I1];
                    Blas<CoefficientPrecision>::axpy(&(n2), &(coef), vv[j].data(), &incx, u1.data(), &incy);
                }

                pivot = 0.;
                tmp   = 0;
                for (int k = 0; k < n2; k++) {
                    if (visited_2[k])
                        continue;
                    tmp = std::abs(u1[k]);
                    if (tmp < pivot)
                        continue;
                    pivot = tmp;
                    I2    = k;
                }
                visited_1[I1]              = true;
                CoefficientPrecision gamma = CoefficientPrecision(1.) / u1[I2];

                //==================//
                // Look for a line
                if (std::abs(u1[I2]) > 1e-15) {
                    std::fill(u2.begin(), u2.end(), CoefficientPrecision(0));
                    if (row_offset >= col_offset) {
                        m_A.copy_submatrix(n1, 1, i1, i2 + I2, u2.data());
                    } else {
                        m_A.copy_submatrix(1, n1, i2 + I2, i1, u2.data());
                    }
                    for (int k = 0; k < uu.size(); k++) {
                        coef = -vv[k][I2];
                        Blas<CoefficientPrecision>::axpy(&(n1), &(coef), uu[k].data(), &incx, u2.data(), &incy);
                    }
                    u2 *= gamma;
                    pivot = 0.;
                    tmp   = 0;
                    for (int k = 0; k < n1; k++) {
                        if (visited_1[k])
                            continue;
                        tmp = std::abs(u2[k]);
                        if (tmp < pivot)
                            continue;
                        pivot = tmp;
                        I1    = k;
                    }
                    visited_2[I2] = true;

                    // Test if no given rank
                    if (reqrank < 0) {
                        // Error estimator
                        CoefficientPrecision frob_aux = 0.;
                        aux                           = std::abs(Blas<CoefficientPrecision>::dot(&(n1), u2.data(), &incx, u2.data(), &incx)) * std::abs(Blas<CoefficientPrecision>::dot(&(n2), u1.data(), &incx, u1.data(), &incx));

                        // aux: terme quadratiques du developpement du carre' de la norme de Frobenius de la matrice low rank
                        for (int j = 0; j < uu.size(); j++) {
                            frob_aux += Blas<CoefficientPrecision>::dot(&(n2), u1.data(), &incx, vv[j].data(), &incy) * Blas<CoefficientPrecision>::dot(&(n1), u2.data(), &(incx), uu[j].data(), &(incy));
                        }
                        // frob_aux: termes croises du developpement du carre' de la norme de Frobenius de la matrice low rank
                        frob += aux + 2 * std::real(frob_aux); // frob: Frobenius norm of the low rank matrix
                                                               //==================//
                    }
                    // Matrix<T> M=A.get_submatrix(this->ir,this->ic);
                    // uu.push_back(M.get_col(J));
                    // vv.push_back(M.get_row(I)/M(I,J));
                    // New cross added
                    uu.push_back(u2);
                    vv.push_back(u1);

                } else {
                    q -= 1;
                    if (q == 0) { // corner case where first row is zero, ACA fails, we build a dense block instead
                        q = -1;
                    }
                    htool::Logger::get_instance().log(LogLevel::WARNING, "ACA found a zero row in a " + std::to_string(M) + "x" + std::to_string(N) + " block. Final rank is " + std::to_string(q)); // LCOV_EXCL_LINE
                    // std::cout << "[Htool warning] ACA found a zero row in a " + std::to_string(M) + "x" + std::to_string(N) + " block. Final rank is " + std::to_string(q) << std::endl;
                    break;
                }
            }
        }

        // Final rank
        rank = q;
        if (rank > 0) {
            auto &U = lrmat.get_U();
            auto &V = lrmat.get_V();
            U.resize(M, rank);
            V.resize(rank, N);

            if (row_offset >= col_offset) {
                for (int k = 0; k < rank; k++) {
                    std::move(uu[k].begin(), uu[k].end(), U.data() + k * M);
                    for (int j = 0; j < N; j++) {
                        V(k, j) = vv[k][j];
                    }
                }
            } else {
                for (int k = 0; k < rank; k++) {
                    std::move(vv[k].begin(), vv[k].end(), U.data() + k * M);
                    for (int j = 0; j < N; j++) {
                        V(k, j) = uu[k][j];
                    }
                }
            }
            return true;
        }
        return false;
    }
};
} // namespace htool
#endif
