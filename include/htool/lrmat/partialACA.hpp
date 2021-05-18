#ifndef HTOOL_PARTIALACA_HPP
#define HTOOL_PARTIALACA_HPP

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
template <typename T, typename ClusterImpl>
class partialACA final : public LowRankMatrix<T, ClusterImpl> {

  public:
    //=========================//
    //    PARTIAL PIVOT ACA    //
    //=========================//
    // If reqrank=-1 (default value), we use the precision given by epsilon for the stopping criterion;
    // otherwise, we use the required rank for the stopping criterion (!: at the end the rank could be lower)
    using LowRankMatrix<T, ClusterImpl>::LowRankMatrix;

    void build(const IMatrix<T> &A, const Cluster<ClusterImpl> &t, const double *const xt, const int *const tabt, const Cluster<ClusterImpl> &s, const double *const xs, const int *const tabs) {
        if (this->rank == 0) {
            this->U.resize(this->nr, 1);
            this->V.resize(1, this->nc);
        } else {

            //// Choice of the first row (see paragraph 3.4.3 page 151 Bebendorf)
            double dist = 1e30;
            int I       = 0;
            for (int i = 0; i < int(this->nr / this->ndofperelt); i++) {
                double aux_dist = std::sqrt(std::inner_product(xt + (t.get_space_dim() * tabt[this->ir[i * this->ndofperelt]]), xt + (t.get_space_dim() * tabt[this->ir[i * this->ndofperelt]]) + t.get_space_dim(), t.get_ctr().begin(), double(0), std::plus<double>(), [](double u, double v) { return (u - v) * (u - v); }));
                if (dist > aux_dist) {
                    dist = aux_dist;
                    I    = i * this->ndofperelt;
                }
            }
            // Partial pivot
            int J       = 0;
            int q       = 0;
            int reqrank = this->rank;
            std::vector<std::vector<T>> uu, vv;
            std::vector<bool> visited_row(this->nr, false);
            std::vector<bool> visited_col(this->nc, false);

            underlying_type<T> frob = 0;
            underlying_type<T> aux  = 0;
            underlying_type<T> pivot, tmp;
            T coef;
            int incx(1), incy(1);
            std::vector<T> r(this->nc), c(this->nr);
            // Either we have a required rank
            // Or it is negative and we have to check the relative error between two iterations.
            //But to do that we need a least two iterations.
            while (((reqrank > 0) && (q < std::min(reqrank, std::min(this->nr, this->nc)))) || ((reqrank < 0) && (q == 0 || sqrt(aux / frob) > this->epsilon))) {
                // if (q != 0)
                // std::cout << sqrt(aux / frob) << " " << this->epsilon << " " << (sqrt(aux / frob) > this->epsilon) << std::endl;
                // Next current rank
                q += 1;

                if (q * (this->nr + this->nc) > (this->nr * this->nc)) { // the next current rank would not be advantageous
                    q = -1;
                    break;
                } else {

                    // Compute the first cross
                    //==================//
                    // Look for a column

                    A.copy_submatrix(1, this->nc, &(this->ir[I]), (this->ic).data(), r.data());
                    for (int j = 0; j < uu.size(); j++) {
                        coef = -uu[j][I];
                        Blas<T>::axpy(&(this->nc), &(coef), vv[j].data(), &incx, r.data(), &incy);
                    }

                    pivot = 0.;
                    tmp   = 0;
                    for (int k = 0; k < this->nc; k++) {
                        if (visited_col[k])
                            continue;
                        tmp = std::abs(r[k]);
                        if (tmp < pivot)
                            continue;
                        pivot = tmp;
                        J     = k;
                        // std::cout << pivot << " " << J << std::endl;
                    }

                    visited_row[I] = true;
                    T gamma        = T(1.) / r[J];
                    //==================//
                    // Look for a line
                    if (std::abs(r[J]) > 1e-15) {

                        A.copy_submatrix(this->nr, 1, (this->ir).data(), &(this->ic[J]), c.data());
                        for (int k = 0; k < uu.size(); k++) {
                            coef = -vv[k][J];
                            Blas<T>::axpy(&(this->nr), &(coef), uu[k].data(), &incx, c.data(), &incy);
                        }
                        c *= gamma;
                        pivot = 0.;
                        tmp   = 0;
                        for (int k = 0; k < this->nr; k++) {
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
                            T frob_aux = 0.;
                            // aux        = std::abs(dprod(c, c) * dprod(r, r));
                            aux = std::abs(Blas<T>::dot(&(this->nr), c.data(), &incx, c.data(), &incx)) * std::abs(Blas<T>::dot(&(this->nc), r.data(), &incx, r.data(), &incx));

                            // aux: terme quadratiques du developpement du carre' de la norme de Frobenius de la matrice low rank
                            for (int j = 0; j < uu.size(); j++) {
                                frob_aux += Blas<T>::dot(&(this->nc), r.data(), &incx, vv[j].data(), &incy) * Blas<T>::dot(&(this->nr), c.data(), &(incx), uu[j].data(), &(incy));
                            }
                            // frob_aux: termes croises du developpement du carre' de la norme de Frobenius de la matrice low rank
                            frob += aux + 2 * std::real(frob_aux); // frob: Frobenius norm of the low rank matrix
                            // std::cout << frob << " " << aux << " " << frob_aux << std::endl;
                            //==================//
                        }
                        // Matrix<T> M=A.get_submatrix(this->ir,this->ic);
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
            this->rank = q;
            if (this->rank > 0) {
                this->U.resize(this->nr, this->rank);
                this->V.resize(this->rank, this->nc);
                for (int k = 0; k < this->rank; k++) {
                    this->U.set_col(k, uu[k]);
                    this->V.set_row(k, vv[k]);
                }
            }
        }
    }
};
} // namespace htool
#endif
