#ifndef HTOOL_FULL_ACA_HPP
#define HTOOL_FULL_ACA_HPP

#include "../../basic_types/vector.hpp"                         // for argmax
#include "../../clustering/cluster_node.hpp"                    // for Cluster
#include "../../hmatrix/interfaces/virtual_generator.hpp"       // for VirtualGenerator
#include "../../hmatrix/interfaces/virtual_lrmat_generator.hpp" // for Virt...
#include "../../matrix/matrix.hpp"                              // for norm...
#include "../../misc/misc.hpp"                                  // for unde...
#include <algorithm>                                            // for min
#include <utility>                                              // for pair
#include <vector>                                               // for vector

namespace htool {

template <typename CoefficientPrecision>
class fullACA final : public VirtualInternalLowRankGenerator<CoefficientPrecision> {

    std::unique_ptr<InternalGeneratorWithPermutation<CoefficientPrecision>> internal_generator_w_permutation;
    const VirtualInternalGenerator<CoefficientPrecision> &m_A;

  public:
    using VirtualInternalLowRankGenerator<CoefficientPrecision>::VirtualInternalLowRankGenerator;

    fullACA(const VirtualInternalGenerator<CoefficientPrecision> &A) : m_A(A) {}
    fullACA(const VirtualGenerator<CoefficientPrecision> &A, const int *target_permutation, const int *source_permutation) : internal_generator_w_permutation(std::make_unique<InternalGeneratorWithPermutation<CoefficientPrecision>>(A, target_permutation, source_permutation)), m_A(*internal_generator_w_permutation) {}

    bool copy_low_rank_approximation(int M, int N, int row_offset, int col_offset, LowRankMatrix<CoefficientPrecision> &lrmat) const override {
        int reqrank = -1;
        return copy_low_rank_approximation(M, N, row_offset, col_offset, lrmat.get_epsilon(), reqrank, lrmat);
    }

    bool copy_low_rank_approximation(int M, int N, int row_offset, int col_offset, int reqrank, LowRankMatrix<CoefficientPrecision> &lrmat) const override {
        return copy_low_rank_approximation(M, N, row_offset, col_offset, lrmat.get_epsilon(), reqrank, lrmat);
    }

  private:
    bool copy_low_rank_approximation(int M, int N, int row_offset, int col_offset, underlying_type<CoefficientPrecision> epsilon, int &rank, LowRankMatrix<CoefficientPrecision> &lrmat) const {
        // Matrix assembling
        Matrix<CoefficientPrecision> mat(M, N);
        m_A.copy_submatrix(M, N, row_offset, col_offset, mat.data());

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
                uu.push_back(get_col(mat, ind.second));
                vv.push_back(get_row(mat, ind.first) / pivot);

                for (int i = 0; i < mat.nb_rows(); i++) {
                    for (int j = 0; j < mat.nb_cols(); j++) {
                        mat(i, j) -= uu[q - 1][i] * vv[q - 1][j];
                    }
                }
            }
        }
        rank = q;
        if (rank > 0) {
            auto &U = lrmat.get_U();
            auto &V = lrmat.get_V();
            U.resize(M, rank);
            V.resize(rank, N);
            for (int k = 0; k < rank; k++) {
                std::move(uu[k].begin(), uu[k].end(), U.data() + k * M);
                for (int j = 0; j < N; j++) {
                    V(k, j) = vv[k][j];
                }
            }
            return true;
        }
        return false;
    }
};

} // namespace htool
#endif
