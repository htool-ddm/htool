#ifndef HTOOL_MULTIPARTIALACA_HPP
#define HTOOL_MULTIPARTIALACA_HPP

#include "../types/multimatrix.hpp"
#include "multilrmat.hpp"
#include <cassert>
#include <complex>
#include <fstream>
#include <iostream>
#include <vector>

namespace htool {

template <typename T>
class MultipartialACA : public VirtualMultiLowRankGenerator<T> {

  public:
    //=========================//
    //    PARTIAL PIVOT ACA    //
    //=========================//
    // If reqrank=-1 (default value), we use the precision given by epsilon for the stopping criterion;
    // otherwise, we use the required rank for the stopping criterion (!: at the end the rank could be lower)
    using VirtualMultiLowRankGenerator<T>::VirtualMultiLowRankGenerator;

    void copy_multi_low_rank_approximation(double epsilon, int M, int N, const int *const rows, const int *const cols, int &rank, T ***U, T ***V, const MultiIMatrix<T> &A, const VirtualCluster &t, const double *const xt, const VirtualCluster &s, const double *const xs) const {

        //// Choice of the first row (see paragraph 3.4.3 page 151 Bebendorf)
        double dist = 1e30;
        int I       = 0;
        for (int i = 0; i < M; i++) {
            double aux_dist = std::sqrt(std::inner_product(xt + (t.get_space_dim() * rows[i]), xt + (t.get_space_dim() * rows[i]) + t.get_space_dim(), t.get_ctr().begin(), double(0), std::plus<double>(), [](double u, double v) { return (u - v) * (u - v); }));

            if (dist > aux_dist) {
                dist = aux_dist;
                I    = i;
            }
        }
        // Partial pivot
        int J       = 0;
        int q       = 0;
        int reqrank = rank;
        std::vector<Matrix<T>> uu, vv;
        std::vector<bool> visited_row(M, false);
        std::vector<bool> visited_col(N, false);

        std::vector<double> frob(A.nb_matrix(), 0);
        std::vector<double> aux(A.nb_matrix(), 0);

        double stopping_criterion = 0;

        // Either we have a required rank
        // Or it is negative and we have to check the relative error between two iterations.
        // But to do that we need a least two iterations.
        while (((reqrank > 0) && (q < std::min(reqrank, std::min(M, N)))) || ((reqrank < 0) && (q == 0 || sqrt(stopping_criterion) > epsilon))) {
            // Next current rank
            q += 1;

            if (q * (M + N) > (M * N)) { // the next current rank would not be advantageous
                q = -1;
                break;
            } else {
                Matrix<T> r(N, A.nb_matrix()), c(M, A.nb_matrix());
                std::vector<T> coefs(A.nb_matrix());

                // Compute the first cross
                //==================//
                // Look for a column
                double pivot = 0.;

                A.copy_submatrices(1, N, &(rows[I]), cols, r.data());

                for (int l = 0; l < A.nb_matrix(); l++) {
                    for (int k = 0; k < N; k++) {
                        r(k, l) = row[l](0, k);
                        for (int j = 0; j < uu.size(); j++) {
                            r(k, l) += -uu[j](I, l) * vv[j](k, l);
                        }
                        if (std::abs(r(k, l)) > pivot && !visited_col[k]) {
                            J     = k;
                            pivot = std::abs(r(k, l));
                        }
                    }
                }

                visited_row[I] = true;
                std::vector<T> gamma(A.nb_matrix());
                for (int l = 0; l < A.nb_matrix(); l++) {
                    gamma[l] = T(1.) / r(J, l);
                }
                //==================//
                // Look for a line
                if (std::abs(min(r.get_row(J))) > 1e-15) {
                    double cmax           = 0.;
                    MultiSubMatrix<T> col = A.get_submatrices(std::vector<int>(rows, rows + M), std::vector<int>{cols[J]});
                    for (int l = 0; l < A.nb_matrix(); l++) {
                        for (int j = 0; j < M; j++) {
                            c(j, l) = col[l](j, 0);
                            for (int k = 0; k < uu.size(); k++) {
                                c(j, l) += -uu[k](j, l) * vv[k](J, l);
                            }
                            c(j, l) = gamma[l] * c(j, l);
                            if (std::abs(c(j, l)) > cmax && !visited_row[j]) {
                                I    = j;
                                cmax = std::abs(c(j, l));
                            }
                        }
                    }

                    visited_col[J] = true;

                    // Test if no given rank
                    if (reqrank < 0) {
                        stopping_criterion = 0;
                        for (int l = 0; l < A.nb_matrix(); l++) {
                            // Error estimator
                            T frob_aux = 0.;
                            aux[l]     = std::abs(dprod(c.get_col(l), c.get_col(l)) * dprod(r.get_col(l), r.get_col(l)));
                            // aux: terme quadratiques du developpement du carre' de la norme de Frobenius de la matrice low rank
                            for (int j = 0; j < uu.size(); j++) {
                                frob_aux += dprod(r.get_col(l), vv[j].get_col(l)) * dprod(c.get_col(l), uu[j].get_col(l));
                            }
                            // frob_aux: termes croises du developpement du carre' de la norme de Frobenius de la matrice low rank
                            frob[l] += aux[l] + 2 * std::real(frob_aux); // frob: Frobenius norm of the low rank matrix
                            //==================//

                            double test = aux[l] / frob[l];
                            if (stopping_criterion < test) {
                                stopping_criterion = test;
                            }
                        }
                    }
                    // Matrix<T> M=A.nb_matrix()get_submatrix(rows,cols);
                    // uu.push_back(M.get_col(J));
                    // vv.push_back(M.get_row(I)/M(I,J));
                    // New cross added
                    uu.push_back(c);
                    vv.push_back(r);

                } else {
                    // std::cout << "There is a zero row in the starting submatrix and ACA didn't work" << std::endl;
                    q -= 1;
                    break;
                }
            }
        }

        // Final rank
        rank = q;
        if (rank > 0) {
            *U = new T *[A.nb_matrix()];
            *V = new T *[A.nb_matrix()];
            for (int l = 0; l < A.nb_matrix(); l++) {
                (*U)[l] = new T[M * rank];
                (*V)[l] = new T[N * rank];
                for (int k = 0; k < rank; k++) {
                    std::copy_n(uu[k].data() + M * l, M, (*U)[l] + k * M);
                    for (int j = 0; j < N; j++) {
                        (*V)[l][rank * j + k] = vv[k](j, k);
                    }
                }
            }
        }
    }
};
} // namespace htool
#endif
