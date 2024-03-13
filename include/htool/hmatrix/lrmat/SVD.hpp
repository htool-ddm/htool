#ifndef HTOOL_SVD_HPP
#define HTOOL_SVD_HPP

#include "../../matrix/utils/SVD_truncation.hpp"
#include "../../wrappers/wrapper_lapack.hpp"
#include "lrmat.hpp"

namespace htool {

template <typename CoefficientPrecision, typename CoordinatesPrecision = underlying_type<CoefficientPrecision>>
class SVD final : public VirtualLowRankGenerator<CoefficientPrecision, CoordinatesPrecision> {

  public:
    using VirtualLowRankGenerator<CoefficientPrecision, CoordinatesPrecision>::VirtualLowRankGenerator;

    void copy_low_rank_approximation(const VirtualGenerator<CoefficientPrecision> &A, const Cluster<CoordinatesPrecision> &target_cluster, const Cluster<CoordinatesPrecision> &source_cluster, underlying_type<CoefficientPrecision> epsilon, int &rank, Matrix<CoefficientPrecision> &U, Matrix<CoefficientPrecision> &V) const override {

        int M = target_cluster.get_size();
        int N = source_cluster.get_size();

        //// Matrix assembling
        Matrix<CoefficientPrecision> mat(M, N);
        A.copy_submatrix(target_cluster.get_size(), source_cluster.get_size(), target_cluster.get_offset(), source_cluster.get_offset(), mat.data());

        //// SVD
        std::vector<underlying_type<CoefficientPrecision>> singular_values(std::min(M, N));
        Matrix<CoefficientPrecision> u(M, M);
        Matrix<CoefficientPrecision> vt(N, N);

        int truncated_rank = SVD_truncation(mat, epsilon, u, vt, singular_values);

        if (rank == -1) {
            if (truncated_rank * (M + N) > (M * N)) {
                truncated_rank = -1;
            }

        } else {
            truncated_rank = std::min(rank, std::min(M, N));
        }

        if (truncated_rank > 0) {
            U.resize(M, truncated_rank);
            V.resize(truncated_rank, N);
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < truncated_rank; j++) {
                    U(i, j) = u(i, j) * singular_values[j];
                }
            }
            for (int i = 0; i < truncated_rank; i++) {
                for (int j = 0; j < N; j++) {
                    V(i, j) = vt(i, j);
                }
            }
        }
    }
};

} // namespace htool

#endif
