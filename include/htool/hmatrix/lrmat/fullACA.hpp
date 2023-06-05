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
template <typename CoefficientPrecision, typename CoordinatesPrecision = underlying_type<CoefficientPrecision>>
class fullACA final : public VirtualLowRankGenerator<CoefficientPrecision, CoordinatesPrecision> {

  public:
    //=========================//
    //    FULL PIVOT ACA    //
    //=========================//
    // If reqrank=-1 (default value), we use the precision given by epsilon for the stopping criterion;
    // otherwise, we use the required rank for the stopping criterion (!: at the end the rank could be lower)
    using VirtualLowRankGenerator<CoefficientPrecision, CoordinatesPrecision>::VirtualLowRankGenerator;

    void copy_low_rank_approximation(const VirtualGenerator<CoefficientPrecision> &A, const Cluster<CoordinatesPrecision> &target_cluster, const Cluster<CoordinatesPrecision> &source_cluster, underlying_type<CoefficientPrecision> epsilon, int &rank, Matrix<CoefficientPrecision> &U, Matrix<CoefficientPrecision> &V) const override {

        int M = target_cluster.get_size();
        int N = source_cluster.get_size();

        // Matrix assembling
        Matrix<CoefficientPrecision> mat(M, N);
        A.copy_submatrix(M, N, target_cluster.get_offset(), source_cluster.get_offset(), mat.data());

        // Full pivot
        int q       = 0;
        int reqrank = rank;
        std::vector<std::vector<CoefficientPrecision>> uu;
        std::vector<std::vector<CoefficientPrecision>> vv;
        double Norm = normFrob(mat);

        while (((reqrank > 0) && (q < std::min(reqrank, std::min(M, N)))) || ((reqrank < 0) && (normFrob(mat) / Norm > epsilon || q == 0))) {

            q += 1;
            if (q * (M + N) > (M * N)) { // the current rank would not be advantageous
                q = -1;
                break;
            } else {
                std::pair<int, int> ind    = argmax(mat);
                CoefficientPrecision pivot = mat(ind.first, ind.second);
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
            U.resize(M, rank);
            V.resize(rank, N);
            for (int k = 0; k < rank; k++) {
                std::move(uu[k].begin(), uu[k].end(), U.data() + k * M);
                for (int j = 0; j < N; j++) {
                    V(k, j) = vv[k][j];
                }
            }
        }
    }
};

} // namespace htool
#endif
