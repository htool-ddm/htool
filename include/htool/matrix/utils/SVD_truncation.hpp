#ifndef HTOOL_MATRIX_UTILS_SVD_TRUNCATION_HPP
#define HTOOL_MATRIX_UTILS_SVD_TRUNCATION_HPP
#include "../../wrappers/wrapper_lapack.hpp"
#include "../matrix.hpp"

namespace htool {

template <typename CoefficientPrecision>
int SVD_truncation(Matrix<CoefficientPrecision> &A, htool::underlying_type<CoefficientPrecision> epsilon, Matrix<CoefficientPrecision> &u, Matrix<CoefficientPrecision> &vt, std::vector<underlying_type<CoefficientPrecision>> &singular_values) {
    // A= u*S*vt
    singular_values.resize(std::min(A.nb_rows(), A.nb_cols()));
    u.resize(A.nb_rows(), A.nb_rows());
    vt.resize(A.nb_cols(), A.nb_cols());
    {
        int M     = A.nb_rows();
        int N     = A.nb_cols();
        int lda   = M;
        int ldu   = M;
        int ldvt  = N;
        int lwork = -1;
        int info;
        std::vector<CoefficientPrecision> work(1);
        std::vector<underlying_type<CoefficientPrecision>> rwork(5 * std::min(M, N));

        Lapack<CoefficientPrecision>::gesvd("A", "A", &M, &N, A.data(), &lda, singular_values.data(), u.data(), &ldu, vt.data(), &ldvt, work.data(), &lwork, rwork.data(), &info);
        lwork = (int)std::real(work[0]);
        work.resize(lwork);
        Lapack<CoefficientPrecision>::gesvd("A", "A", &M, &N, A.data(), &lda, singular_values.data(), u.data(), &ldu, vt.data(), &ldvt, work.data(), &lwork, rwork.data(), &info);
    }

    // Compute truncated rank to define tildeS
    int truncated_rank;
    {
        int j                                          = singular_values.size();
        underlying_type<CoefficientPrecision> svd_norm = 0;
        underlying_type<CoefficientPrecision> error    = 0;
        for (auto &elt : singular_values)
            svd_norm += elt * elt;
        svd_norm = std::sqrt(svd_norm);

        do {
            j = j - 1;
            error += std::pow(std::abs(singular_values[j]), 2);
        } while (j > 0 && std::sqrt(error) / svd_norm < epsilon);

        truncated_rank = j + 1;
    }

    return truncated_rank;
}
} // namespace htool
#endif
