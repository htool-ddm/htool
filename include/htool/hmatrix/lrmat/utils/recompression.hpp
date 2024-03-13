#ifndef HTOOL_LRMAT_UTILS_RECOMPRESSION_HPP
#define HTOOL_LRMAT_UTILS_RECOMPRESSION_HPP
#include "../../../matrix/linalg/add_matrix_matrix_product.hpp"
#include "../../../matrix/utils/SVD_truncation.hpp"
#include "../../../wrappers/wrapper_lapack.hpp"
#include "../lrmat.hpp"

namespace htool {

template <typename CoefficientPrecision>
void recompression(LowRankMatrix<CoefficientPrecision> &lrmat) {
    Matrix<CoefficientPrecision> U = lrmat.get_U();
    Matrix<CoefficientPrecision> V = lrmat.get_V();
    auto rank                      = lrmat.rank_of();
    auto epsilon                   = lrmat.get_epsilon();

    if (rank > std::min(U.nb_rows(), V.nb_cols())) {
        Matrix<CoefficientPrecision> UV(U.nb_rows(), V.nb_cols());
        add_matrix_matrix_product('N', 'N', CoefficientPrecision(1), U, V, CoefficientPrecision(1), UV);

        // UV= u*S*vt
        std::vector<underlying_type<CoefficientPrecision>> singular_values(std::min(U.nb_rows(), V.nb_cols()));
        Matrix<CoefficientPrecision> u(U.nb_rows(), U.nb_rows());
        Matrix<CoefficientPrecision> vt(V.nb_cols(), V.nb_cols());
        int truncated_rank = SVD_truncation(UV, epsilon, u, vt, singular_values);

        // new_U=u*sqrt(tildeS) and new_V=sqrt(tildeS)*vt in the right dimensions
        Matrix<CoefficientPrecision> &new_U = lrmat.get_U();
        Matrix<CoefficientPrecision> &new_V = lrmat.get_V();
        {
            int M    = U.nb_rows();
            int N    = V.nb_cols();
            int incx = 1;
            new_U.resize(M, truncated_rank);
            new_V.resize(truncated_rank, N);
            CoefficientPrecision scaling_coef;
            for (int r = 0; r < truncated_rank; r++) {
                scaling_coef = std::sqrt(singular_values[r]);
                std::copy_n(u.data() + r * u.nb_rows(), u.nb_rows(), new_U.data() + r * M);
                Blas<CoefficientPrecision>::scal(&M, &scaling_coef, new_U.data() + r * M, &incx);
            }
            for (int r = 0; r < vt.nb_cols(); r++) {
                std::copy_n(vt.data() + r * vt.nb_rows(), truncated_rank, new_V.data() + r * truncated_rank);
            }

            for (int r = 0; r < truncated_rank; r++) {
                for (int j = 0; j < vt.nb_cols(); j++) {
                    new_V(r, j) = std::sqrt(singular_values[r]) * new_V(r, j);
                }
            }
        }

    } else {
        // U=Q1R
        std::vector<CoefficientPrecision> tau_QR;
        {
            int M     = U.nb_rows();
            int N     = U.nb_cols();
            int lda   = U.nb_rows();
            int lwork = -1;
            int info;
            std::vector<CoefficientPrecision> work(1);
            tau_QR.resize(std::min(M, N));
            Lapack<CoefficientPrecision>::geqrf(&M, &N, U.data(), &lda, tau_QR.data(), work.data(), &lwork, &info);
            lwork = (int)std::real(work[0]);
            work.resize(lwork);
            Lapack<CoefficientPrecision>::geqrf(&M, &N, U.data(), &lda, tau_QR.data(), work.data(), &lwork, &info);
        }

        // V=LQ2
        std::vector<CoefficientPrecision> tau_LQ;
        {
            int M     = V.nb_rows();
            int N     = V.nb_cols();
            int lda   = V.nb_rows();
            int lwork = -1;
            int info;
            std::vector<CoefficientPrecision> work(1);
            tau_LQ.resize(std::min(M, N));
            Lapack<CoefficientPrecision>::gelqf(&M, &N, V.data(), &lda, tau_LQ.data(), work.data(), &lwork, &info);
            lwork = (int)std::real(work[0]);
            work.resize(lwork);
            Lapack<CoefficientPrecision>::gelqf(&M, &N, V.data(), &lda, tau_LQ.data(), work.data(), &lwork, &info);
        }

        // RL = R*L
        Matrix<CoefficientPrecision> RL(rank, rank);
        {
            Matrix<CoefficientPrecision> R(rank, rank), L(rank, rank);
            for (int j = 0; j < rank; j++) {
                for (int i = 0; i <= j; i++) {
                    R(i, j) = U(i, j);
                }
            }
            for (int j = 0; j < rank; j++) {
                for (int i = j; i < rank; i++) {
                    L(i, j) = V(i, j);
                }
            }

            add_matrix_matrix_product('N', 'N', CoefficientPrecision(1), R, L, CoefficientPrecision(0), RL);
        }

        // RL= u*S*vt
        std::vector<underlying_type<CoefficientPrecision>> singular_values(rank);
        Matrix<CoefficientPrecision> u(rank, rank);
        Matrix<CoefficientPrecision> vt(rank, rank);
        int truncated_rank = SVD_truncation(RL, epsilon, u, vt, singular_values);

        // new_U=u*sqrt(tildeS) and new_V=sqrt(tildeS)*vt in the right dimensions
        Matrix<CoefficientPrecision> &new_U = lrmat.get_U();
        Matrix<CoefficientPrecision> &new_V = lrmat.get_V();
        {
            int M    = U.nb_rows();
            int N    = V.nb_cols();
            int incx = 1;
            new_U.resize(M, truncated_rank);
            new_V.resize(truncated_rank, N);
            CoefficientPrecision scaling_coef;
            for (int r = 0; r < truncated_rank; r++) {
                scaling_coef = std::sqrt(singular_values[r]);
                std::copy_n(u.data() + r * rank, rank, new_U.data() + r * M);
                Blas<CoefficientPrecision>::scal(&M, &scaling_coef, new_U.data() + r * M, &incx);
            }
            for (int r = 0; r < rank; r++) {
                std::copy_n(vt.data() + r * rank, truncated_rank, new_V.data() + r * truncated_rank);
            }

            for (int r = 0; r < truncated_rank; r++) {
                for (int j = 0; j < rank; j++) {
                    new_V(r, j) = std::sqrt(singular_values[r]) * new_V(r, j);
                }
            }
        }

        // new_U=Q1*new_U
        {
            char size  = 'L';
            char trans = 'N';
            int M      = new_U.nb_rows();
            int N      = new_U.nb_cols();
            int K      = tau_QR.size();
            int ldc    = M;
            int lwork  = -1;
            int info;
            std::vector<CoefficientPrecision> work(1);
            Lapack<CoefficientPrecision>::mqr(&size, &trans, &M, &N, &K, U.data(), &M, tau_QR.data(), new_U.data(), &ldc, work.data(), &lwork, &info);
            lwork = (int)std::real(work[0]);
            work.resize(lwork);
            Lapack<CoefficientPrecision>::mqr(&size, &trans, &M, &N, &K, U.data(), &M, tau_QR.data(), new_U.data(), &ldc, work.data(), &lwork, &info);
        }

        // new_V=new_V*Q2
        {
            char size  = 'R';
            char trans = 'N';
            int M      = new_V.nb_rows();
            int N      = new_V.nb_cols();
            int K      = tau_LQ.size();
            int lda    = V.nb_rows();
            int ldc    = M;
            int lwork  = -1;
            int info;
            std::vector<CoefficientPrecision> work(1);
            Lapack<CoefficientPrecision>::mlq(&size, &trans, &M, &N, &K, V.data(), &lda, tau_LQ.data(), new_V.data(), &ldc, work.data(), &lwork, &info);
            lwork = (int)std::real(work[0]);
            work.resize(lwork);
            Lapack<CoefficientPrecision>::mlq(&size, &trans, &M, &N, &K, V.data(), &lda, tau_LQ.data(), new_V.data(), &ldc, work.data(), &lwork, &info);
        }
    }
}

template <typename CoefficientPrecision>
std::unique_ptr<LowRankMatrix<CoefficientPrecision>> recompression(const LowRankMatrix<CoefficientPrecision> &lrmat) {
    std::unique_ptr<LowRankMatrix<CoefficientPrecision>> new_lrmat_ptr = std::make_unique<LowRankMatrix<CoefficientPrecision>>(lrmat);
    recompression(*new_lrmat_ptr);
    return new_lrmat_ptr;
}

} // namespace htool

#endif
