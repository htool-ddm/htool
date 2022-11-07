#ifndef HTOOL_SYMPARTIALACA_HPP
#define HTOOL_SYMPARTIALACA_HPP

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
template <typename T>
class sympartialACA final : public VirtualLowRankGenerator<T> {

  public:
    //=========================//
    //    PARTIAL PIVOT ACA    //
    //=========================//
    // If reqrank=-1 (default value), we use the precision given by epsilon for the stopping criterion;
    // otherwise, we use the required rank for the stopping criterion (!: at the end the rank could be lower)
    using VirtualLowRankGenerator<T>::VirtualLowRankGenerator;

    void copy_low_rank_approximation(double epsilon, int M, int N, const int *const rows, const int *const cols, int &rank, T **U, T **V, const VirtualGenerator<T> &A, const VirtualCluster &t, const double *const xt, const VirtualCluster &s, const double *const xs) const {

        int n1, n2;
        const int *i1;
        const int *i2;
        const double *x1;
        VirtualCluster const *cluster_1;

        if (t.get_offset() >= s.get_offset()) {

            n1        = M;
            n2        = N;
            i1        = rows;
            i2        = cols;
            x1        = xt;
            cluster_1 = &t;
        } else {
            n1        = N;
            n2        = M;
            i1        = cols;
            i2        = rows;
            x1        = xs;
            cluster_1 = &s;
        }

        //// Choice of the first row (see paragraph 3.4.3 page 151 Bebendorf)
        double dist = 1e30;
        int I1      = 0;
        for (int i = 0; i < n1; i++) {
            double aux_dist = std::sqrt(std::inner_product(x1 + (cluster_1->get_space_dim() * i1[i]), x1 + (cluster_1->get_space_dim() * i1[i]) + cluster_1->get_space_dim(), cluster_1->get_ctr().begin(), double(0), std::plus<double>(), [](double u, double v) { return (u - v) * (u - v); }));

            if (dist > aux_dist) {
                dist = aux_dist;
                I1   = i;
            }
        }
        // Partial pivot
        int I2      = 0;
        int q       = 0;
        int reqrank = rank;
        std::vector<std::vector<T>> uu, vv;
        std::vector<bool> visited_1(n1, false);
        std::vector<bool> visited_2(n2, false);

        underlying_type<T> frob = 0;
        underlying_type<T> aux  = 0;

        underlying_type<T> pivot, tmp;
        T coef;
        int incx(1), incy(1);
        std::vector<T> u1(n2), u2(n1);

        // Either we have a required rank
        // Or it is negative and we have to check the relative error between two iterations.
        // But to do that we need a least two iterations.
        while (((reqrank > 0) && (q < std::min(reqrank, std::min(M, N)))) || ((reqrank < 0) && (sqrt(aux / frob) > epsilon || q == 0))) {

            // Next current rank
            q += 1;

            if (q * (M + N) > (M * N)) { // the next current rank would not be advantageous
                q = -1;
                break;
            } else {

                if (t.get_offset() >= s.get_offset()) {
                    A.copy_submatrix(1, n2, &(i1[I1]), i2, u1.data());
                } else {
                    A.copy_submatrix(n2, 1, i2, &(i1[I1]), u1.data());
                }

                for (int j = 0; j < uu.size(); j++) {
                    coef = -uu[j][I1];
                    Blas<T>::axpy(&(n2), &(coef), vv[j].data(), &incx, u1.data(), &incy);
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
                visited_1[I1] = true;
                T gamma       = T(1.) / u1[I2];

                //==================//
                // Look for a line
                if (std::abs(u1[I2]) > 1e-15) {
                    if (t.get_offset() >= s.get_offset()) {
                        A.copy_submatrix(n1, 1, i1, &(i2[I2]), u2.data());
                    } else {
                        A.copy_submatrix(1, n1, &(i2[I2]), i1, u2.data());
                    }
                    for (int k = 0; k < uu.size(); k++) {
                        coef = -vv[k][I2];
                        Blas<T>::axpy(&(n1), &(coef), uu[k].data(), &incx, u2.data(), &incy);
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
                        T frob_aux = 0.;
                        aux        = std::abs(Blas<T>::dot(&(n1), u2.data(), &incx, u2.data(), &incx)) * std::abs(Blas<T>::dot(&(n2), u1.data(), &incx, u1.data(), &incx));

                        // aux: terme quadratiques du developpement du carre' de la norme de Frobenius de la matrice low rank
                        for (int j = 0; j < uu.size(); j++) {
                            frob_aux += Blas<T>::dot(&(n2), u1.data(), &incx, vv[j].data(), &incy) * Blas<T>::dot(&(n1), u2.data(), &(incx), uu[j].data(), &(incy));
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
                    std::cout << "[Htool warning] ACA found a zero row in a " + std::to_string(M) + "x" + std::to_string(N) + " block. Final rank is " + std::to_string(q) << std::endl;
                    break;
                }
            }
        }

        // Final rank
        rank = q;
        if (rank > 0) {
            *U = new T[M * rank];
            *V = new T[rank * N];
            for (int k = 0; k < rank; k++) {
                if (t.get_offset() >= s.get_offset()) {
                    for (int k = 0; k < rank; k++) {
                        std::copy_n(uu[k].begin(), uu[k].size(), *U + k * M);
                        for (int j = 0; j < N; j++) {
                            (*V)[rank * j + k] = vv[k][j];
                        }
                    }
                } else {
                    for (int k = 0; k < rank; k++) {
                        std::copy_n(vv[k].begin(), vv[k].size(), *U + k * M);
                        for (int j = 0; j < N; j++) {
                            (*V)[rank * j + k] = uu[k][j];
                        }
                    }
                }
            }
        }
    }
};
} // namespace htool
#endif
