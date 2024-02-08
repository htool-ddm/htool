#ifndef HTOOL_SVD_HPP
#define HTOOL_SVD_HPP

#include "../../wrappers/wrapper_lapack.hpp"
#include "lrmat.hpp"

namespace htool {

template <typename CoefficientPrecision, typename CoordinatesPrecision = underlying_type<CoefficientPrecision>>
class SVD final : public VirtualLowRankGenerator<CoefficientPrecision, CoordinatesPrecision> {

  public:
    using VirtualLowRankGenerator<CoefficientPrecision, CoordinatesPrecision>::VirtualLowRankGenerator;

    void copy_low_rank_approximation(const VirtualGenerator<CoefficientPrecision> &A, const Cluster<CoordinatesPrecision> &target_cluster, const Cluster<CoordinatesPrecision> &source_cluster, underlying_type<CoefficientPrecision> epsilon, int &rank, Matrix<CoefficientPrecision> &U, Matrix<CoefficientPrecision> &V) const override {

        int reqrank = 0;
        int M       = target_cluster.get_size();
        int N       = source_cluster.get_size();

        //// Matrix assembling
        double Norm = 0;
        Matrix<CoefficientPrecision> mat(M, N);
        A.copy_submatrix(target_cluster.get_size(), source_cluster.get_size(), target_cluster.get_offset(), source_cluster.get_offset(), mat.data());

        Norm = normFrob(mat);

        //// SVD
        int m     = M;
        int n     = N;
        int lda   = m;
        int ldu   = m;
        int ldvt  = n;
        int lwork = -1;
        int info;
        std::vector<underlying_type<CoefficientPrecision>> singular_values(std::min(m, n));
        Matrix<CoefficientPrecision> u(m, m);
        // std::vector<CoefficientPrecision> vt (n*n);
        Matrix<CoefficientPrecision> vt(n, n);
        std::vector<CoefficientPrecision> work(std::min(m, n));
        std::vector<underlying_type<CoefficientPrecision>> rwork(5 * std::min(m, n));

        Lapack<CoefficientPrecision>::gesvd("A", "A", &m, &n, mat.data(), &lda, singular_values.data(), u.data(), &ldu, vt.data(), &ldvt, work.data(), &lwork, rwork.data(), &info);
        lwork = (int)std::real(work[0]);
        work.resize(lwork);
        Lapack<CoefficientPrecision>::gesvd("A", "A", &m, &n, mat.data(), &lda, singular_values.data(), u.data(), &ldu, vt.data(), &ldvt, work.data(), &lwork, rwork.data(), &info);

        if (rank == -1) {

            // Compute Frobenius norm of the approximation error
            int j           = singular_values.size();
            double svd_norm = 0;

            do {
                j = j - 1;
                svd_norm += std::pow(std::abs(singular_values[j]), 2);
            } while (j > 0 && std::sqrt(svd_norm) / Norm < epsilon);

            reqrank = std::min(j + 1, std::min(m, n));

            if (reqrank * (M + N) > (M * N)) {
                reqrank = -1;
            }
            rank = reqrank;

        } else {
            reqrank = std::min(rank, std::min(M, N));
        }

        if (rank > 0) {
            U.resize(M, rank);
            V.resize(rank, N);
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < reqrank; j++) {
                    U(i, j) = u(i, j) * singular_values[j];
                }
            }
            for (int i = 0; i < reqrank; i++) {
                for (int j = 0; j < N; j++) {
                    V(i, j) = vt(i, j);
                }
            }
        }
    }

    // T get_singular_value(int i) { return singular_values[i]; }
};

} // namespace htool

#endif
