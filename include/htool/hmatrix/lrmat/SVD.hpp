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

template <typename CoefficientPrecision, typename CoordinatesPrecision = underlying_type<CoefficientPrecision>>
class SVD final : public VirtualInternalLowRankGenerator<CoefficientPrecision> {

    const VirtualInternalGenerator<CoefficientPrecision> &m_A;

  public:
    using VirtualInternalLowRankGenerator<CoefficientPrecision>::VirtualInternalLowRankGenerator;

    SVD(const VirtualInternalGenerator<CoefficientPrecision> &A) : m_A(A) {}
    SVD(const VirtualGenerator<CoefficientPrecision> &A) : m_A(InternalGeneratorWithPermutation<CoefficientPrecision>(A)) {}

    void copy_low_rank_approximation(int M, int N, int row_offset, int col_offset, underlying_type<CoefficientPrecision> epsilon, int &rank, Matrix<CoefficientPrecision> &U, Matrix<CoefficientPrecision> &V) const override {

        //// Matrix assembling
        Matrix<CoefficientPrecision> mat(M, N);
        m_A.copy_submatrix(M, N, row_offset, col_offset, mat.data());

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
