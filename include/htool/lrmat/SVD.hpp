#ifndef HTOOL_SVD_HPP
#define HTOOL_SVD_HPP

#include "../wrappers/wrapper_lapack.hpp"
#include "lrmat.hpp"

namespace htool {

template <typename T>
class SVD final : public VirtualLowRankGenerator<T> {

  public:
    using VirtualLowRankGenerator<T>::VirtualLowRankGenerator;

    void copy_low_rank_approximation(double epsilon, int M, int N, const int *const rows, const int *const cols, int &rank, T **U, T **V, const VirtualGenerator<T> &A, const VirtualCluster &, const double *const, const VirtualCluster &, const double *const) const {
        int reqrank = 0;
        //// Matrix assembling
        double Norm = 0;
        std::vector<T> mat(M * N);
        A.copy_submatrix(M, N, rows, cols, mat.data());
        for (int i = 0; i < mat.size(); i++) {
            Norm += std::abs(mat[i] * mat[i]);
        }
        Norm = sqrt(Norm);

        //// SVD
        int m     = M;
        int n     = N;
        int lda   = m;
        int ldu   = m;
        int ldvt  = n;
        int lwork = -1;
        int info;
        std::vector<underlying_type<T>> singular_values(std::min(m, n));
        Matrix<T> u(m, m);
        // std::vector<T> vt (n*n);
        Matrix<T> vt(n, n);
        std::vector<T> work(std::min(m, n));
        std::vector<underlying_type<T>> rwork(5 * std::min(m, n));

        Lapack<T>::gesvd("A", "A", &m, &n, mat.data(), &lda, singular_values.data(), u.data(), &ldu, vt.data(), &ldvt, work.data(), &lwork, rwork.data(), &info);
        lwork = (int)std::real(work[0]);
        work.resize(lwork);
        Lapack<T>::gesvd("A", "A", &m, &n, mat.data(), &lda, singular_values.data(), u.data(), &ldu, vt.data(), &ldvt, work.data(), &lwork, rwork.data(), &info);

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
            *U = new T[M * rank];
            *V = new T[rank * N];
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < reqrank; j++) {
                    (*U)[i + M * j] = u(i, j) * singular_values[j];
                }
            }
            for (int i = 0; i < reqrank; i++) {
                for (int j = 0; j < N; j++) {
                    (*V)[i + rank * j] = vt(i, j);
                }
            }
        }
    }

    // T get_singular_value(int i) { return singular_values[i]; }
};

} // namespace htool

#endif
