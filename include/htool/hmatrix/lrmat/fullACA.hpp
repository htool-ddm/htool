#ifndef HTOOL_FULL_ACA_HPP
#define HTOOL_FULL_ACA_HPP

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
class fullACA final : public VirtualLowRankGenerator<T> {

  public:
    //=========================//
    //    FULL PIVOT ACA    //
    //=========================//
    // If reqrank=-1 (default value), we use the precision given by epsilon for the stopping criterion;
    // otherwise, we use the required rank for the stopping criterion (!: at the end the rank could be lower)
    using VirtualLowRankGenerator<T>::VirtualLowRankGenerator;

    void copy_low_rank_approximation(double epsilon, int M, int N, const int *const rows, const int *const cols, int &rank, T **U, T **V, const VirtualGenerator<T> &A, const VirtualCluster &, const double *const, const VirtualCluster &, const double *const) const {

        // Matrix assembling
        Matrix<T> mat(M, N);
        A.copy_submatrix(M, N, rows, cols, mat.data());

        // Full pivot
        int q       = 0;
        int reqrank = rank;
        std::vector<std::vector<T>> uu;
        std::vector<std::vector<T>> vv;
        double Norm = normFrob(mat);

        while (((reqrank > 0) && (q < std::min(reqrank, std::min(M, N)))) || ((reqrank < 0) && (normFrob(mat) / Norm > epsilon || q == 0))) {

            q += 1;
            if (q * (M + N) > (M * N)) { // the current rank would not be advantageous
                q = -1;
                break;
            } else {
                std::pair<int, int> ind = argmax(mat);
                T pivot                 = mat(ind.first, ind.second);
                if (std::abs(pivot) < 1e-15) {
                    q += -1;
                    break;
                }
                uu.push_back(mat.get_col(ind.second));
                vv.push_back(mat.get_row(ind.first) / pivot);

                for (int i = 0; i < mat.nb_rows(); i++) {
                    for (int j = 0; j < mat.nb_cols(); j++) {
                        mat(i, j) -= uu[q - 1][i] * vv[q - 1][j];
                    }
                }
            }
        }
        rank = q;
        if (rank > 0) {
            *U = new T[M * rank];
            *V = new T[rank * N];
            for (int k = 0; k < rank; k++) {
                std::copy_n(uu[k].begin(), uu[k].size(), *U + k * M);
                for (int j = 0; j < N; j++) {
                    (*V)[rank * j + k] = vv[k][j];
                }
            }
        }
    }
};

} // namespace htool
#endif
