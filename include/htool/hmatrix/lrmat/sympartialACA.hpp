#ifndef HTOOL_SYMPARTIALACA_HPP
#define HTOOL_SYMPARTIALACA_HPP

#include "../../misc/logger.hpp"
#include "lrmat.hpp"
#include <vector>

namespace htool {
//================================//
//   CLASSE MATRICE RANG FAIBLE   //
//================================//
//
// Refs biblio:
//
//  -> slides de Stéphanie Chaillat:
//           http://uma.ensta-paristech.fr/var/files/chaillat/seance2.pdf
//           et en particulier la slide 25
//
//  -> livre de M.Bebendorf:
//           http://www.springer.com/kr/book/9783540771463
//           et en particulier le paragraphe 3.4
//
//  -> livre de Rjasanow-Steinbach:
//           http://www.ems-ph.org/books/book.php?proj_nr=125
//           et en particulier le paragraphe 3.2
//
//=================================//
template <typename CoefficientPrecision, typename CoordinatePrecision = underlying_type<CoefficientPrecision>>
class sympartialACA final : public VirtualLowRankGenerator<CoefficientPrecision, CoordinatePrecision> {

  public:
    //=========================//
    //    PARTIAL PIVOT ACA    //
    //=========================//
    // If reqrank=-1 (default value), we use the precision given by epsilon for the stopping criterion;
    // otherwise, we use the required rank for the stopping criterion (!: at the end the rank could be lower)
    using VirtualLowRankGenerator<CoefficientPrecision, CoordinatePrecision>::VirtualLowRankGenerator;

    void copy_low_rank_approximation(const VirtualGenerator<CoefficientPrecision> &A, const Cluster<CoordinatePrecision> &t, const Cluster<CoordinatePrecision> &s, double epsilon, int &rank, Matrix<CoefficientPrecision> &U, Matrix<CoefficientPrecision> &V) const {

        int n1, n2;
        int i1;
        int i2;
        // const double *x1;

        if (t.get_offset() >= s.get_offset()) {

            n1 = t.get_size();
            n2 = s.get_size();
            i1 = t.get_offset();
            i2 = s.get_offset();
            // x1        = xt;
            // cluster_1 = &t;
        } else {
            n1 = s.get_size();
            n2 = t.get_size();
            i1 = s.get_offset();
            i2 = t.get_offset();
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
                if (t.get_offset() >= s.get_offset()) {
                    A.copy_submatrix(1, n2, i1 + I1, i2, u1.data());
                } else {
                    A.copy_submatrix(n2, 1, i2, i1 + I1, u1.data());
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
                    if (t.get_offset() >= s.get_offset()) {
                        A.copy_submatrix(n1, 1, i1, i2 + I2, u2.data());
                    } else {
                        A.copy_submatrix(1, n1, i2 + I2, i1, u2.data());
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
                    htool::Logger::get_instance().log(LogLevel::WARNING, "ACA found a zero row in a " + std::to_string(t.get_size()) + "x" + std::to_string(s.get_size()) + " block. Final rank is " + std::to_string(q)); // LCOV_EXCL_LINE
                    // std::cout << "[Htool warning] ACA found a zero row in a " + std::to_string(t.get_size()) + "x" + std::to_string(s.get_size()) + " block. Final rank is " + std::to_string(q) << std::endl;
                    break;
                }
            }
        }

        // Final rank
        rank = q;
        if (rank > 0) {
            U.resize(t.get_size(), rank);
            V.resize(rank, s.get_size());

            if (t.get_offset() >= s.get_offset()) {
                for (int k = 0; k < rank; k++) {
                    std::move(uu[k].begin(), uu[k].end(), U.data() + k * t.get_size());
                    for (int j = 0; j < s.get_size(); j++) {
                        V(k, j) = vv[k][j];
                    }
                }
            } else {
                for (int k = 0; k < rank; k++) {
                    std::move(vv[k].begin(), vv[k].end(), U.data() + k * t.get_size());
                    for (int j = 0; j < s.get_size(); j++) {
                        V(k, j) = uu[k][j];
                    }
                }
            }
        }
    }
};
} // namespace htool
#endif
