#ifndef HTOOL_SVD_HPP
#define HTOOL_SVD_HPP

#include "../../clustering/cluster_node.hpp"                    // for Cluster
#include "../../hmatrix/interfaces/virtual_generator.hpp"       // for VirtualGenerator
#include "../../hmatrix/interfaces/virtual_lrmat_generator.hpp" // for Virt...
#include "../../matrix/matrix.hpp"                              // for Matrix
#include "../../matrix/utils/SVD_truncation.hpp"                // for SVD_...
#include "../../misc/misc.hpp"                                  // for unde...
#include <algorithm>                                            // for min
#include <vector>                                               // for vector

namespace htool {

template <typename CoefficientPrecision>
class SVD final : public VirtualInternalLowRankGenerator<CoefficientPrecision> {

    std::unique_ptr<InternalGeneratorWithPermutation<CoefficientPrecision>> internal_generator_w_permutation;
    const VirtualInternalGenerator<CoefficientPrecision> &m_A;

  public:
    using VirtualInternalLowRankGenerator<CoefficientPrecision>::VirtualInternalLowRankGenerator;

    SVD(const VirtualInternalGenerator<CoefficientPrecision> &A) : m_A(A) {}
    SVD(const VirtualGenerator<CoefficientPrecision> &A, const int *target_permutation, const int *source_permutation) : internal_generator_w_permutation(std::make_unique<InternalGeneratorWithPermutation<CoefficientPrecision>>(A, target_permutation, source_permutation)), m_A(*internal_generator_w_permutation) {}

    bool copy_low_rank_approximation(int M, int N, int row_offset, int col_offset, LowRankMatrix<CoefficientPrecision> &lrmat) const override {

        //// Matrix assembling
        Matrix<CoefficientPrecision> mat(M, N);
        m_A.copy_submatrix(M, N, row_offset, col_offset, mat.data());

        //// SVD
        std::vector<underlying_type<CoefficientPrecision>> singular_values(std::min(M, N));
        Matrix<CoefficientPrecision> u(M, M);
        Matrix<CoefficientPrecision> vt(N, N);

        int truncated_rank = SVD_truncation(mat, lrmat.get_epsilon(), u, vt, singular_values);

        if (truncated_rank * (M + N) > (M * N)) {
            return false;
        }

        if (truncated_rank > 0) {
            auto &U = lrmat.get_U();
            auto &V = lrmat.get_V();
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
            return true;
        }
        return false;
    }

    bool copy_low_rank_approximation(int M, int N, int row_offset, int col_offset, int rank, LowRankMatrix<CoefficientPrecision> &lrmat) const override {
        //// Matrix assembling
        Matrix<CoefficientPrecision> mat(M, N);
        m_A.copy_submatrix(M, N, row_offset, col_offset, mat.data());

        //// SVD
        std::vector<underlying_type<CoefficientPrecision>> singular_values(std::min(M, N));
        Matrix<CoefficientPrecision> u(M, M);
        Matrix<CoefficientPrecision> vt(N, N);

        int truncated_rank = SVD_truncation(mat, lrmat.get_epsilon(), u, vt, singular_values);
        truncated_rank     = std::min(rank, std::min(M, N));

        auto &U = lrmat.get_U();
        auto &V = lrmat.get_V();
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
        return true;
    }
};

} // namespace htool

#endif
