#ifndef HTOOL_VIRTUAL_IBCOMP_HPP
#define HTOOL_VIRTUAL_IBCOMP_HPP

#include "../../hmatrix/interfaces/virtual_lrmat_generator.hpp" // for Virt...
#include "../../wrappers/wrapper_blas.hpp"                      // for Blas
#include "../../wrappers/wrapper_lapack.hpp"                    // for Blas
#include "lrmat.hpp"
#include <functional>
#include <map>
#include <span>

#if defined(__clang__)
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wold-style-cast"
#elif defined(__GNUC__) || defined(__GNUG__)
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wold-style-cast"
#endif

#include "theia.hpp"

#if defined(__clang__)
#    pragma clang diagnostic pop
#elif defined(__GNUC__) || defined(__GNUG__)
#    pragma GCC diagnostic pop
#endif

namespace htool {

template <typename CoefficientPrecision, typename CoordinatePrecision, int dimension>
class HCA final : public VirtualInternalLowRankGenerator<CoefficientPrecision> {

  private:
    std::function<void(std::array<CoordinatePrecision, dimension> *, int, std::array<CoordinatePrecision, dimension> *, int, CoefficientPrecision *)> m_kernel;
    const int *m_target_permutation;
    const int *m_source_permutation;
    const CoordinatePrecision *m_target_points;
    const CoordinatePrecision *m_source_points;
    // const std::map<int, std::vector<std::pair<int, CoordinatePrecision>>> &
    //     m_target_dof_to_elts;
    // const std::map<int, std::vector<std::pair<int, CoordinatePrecision>>> &m_source_dof_to_elts;

  public:
    // HCA(std::function<CoefficientPrecision(std::array<CoordinatePrecision, 3>, std::array<CoordinatePrecision, 3>)> kernel, const int *target_permutation, const std::map<int, std::vector<std::pair<int, CoordinatePrecision>>> &target_dof_to_elts, const int *source_permutation, const std::map<int, std::vector<std::pair<int, CoordinatePrecision>>> &source_dof_to_elts) : m_kernel(kernel), m_target_permutation(target_permutation), m_target_dof_to_elts(target_dof_to_elts), m_source_permutation(source_permutation), m_source_dof_to_elts(source_dof_to_elts) {}

    HCA(std::function<void(std::array<CoordinatePrecision, dimension> *, int, std::array<CoordinatePrecision, dimension> *, int, CoefficientPrecision *)> kernel, const int *target_permutation, const int *source_permutation) : m_kernel(kernel), m_target_permutation(target_permutation), m_source_permutation(source_permutation) {}

    virtual bool copy_low_rank_approximation(int M, int N, int row_offset, int col_offset, LowRankMatrix<CoefficientPrecision> &lrmat) const override {
        return copy_low_rank_approximation(M, N, m_target_permutation + row_offset, m_source_permutation + col_offset, lrmat);
    }

    virtual bool copy_low_rank_approximation(int M, int N, int row_offset, int col_offset, int reqrank, LowRankMatrix<CoefficientPrecision> &lrmat) const override {
        return copy_low_rank_approximation(M, N, m_target_permutation + row_offset, m_source_permutation + col_offset, reqrank, lrmat);
    }

    bool copy_low_rank_approximation(int M, int N, const int *rows, const int *cols, LowRankMatrix<CoefficientPrecision> &lrmat) const {
        std::array<CoordinatePrecision, 2 * dimension> target_box;
        std::array<CoordinatePrecision, 2 * dimension> source_box;
        for (int dim = 0; dim < dimension; dim++) {
            target_box[2 * dim + 0] = std::numeric_limits<double>::min();
            target_box[2 * dim + 1] = std::numeric_limits<double>::max();
            source_box[2 * dim + 0] = std::numeric_limits<double>::min();
            source_box[2 * dim + 1] = std::numeric_limits<double>::max();
        }
        for (int i = 0; i < M; i++) {
            for (int dim = 0; dim < dimension; dim++) {
                target_box[2 * dim + 0] = std::max(target_box[2 * dim + 0], m_target_points[rows[i] * dim + dim]);
                target_box[2 * dim + 1] = std::min(target_box[2 * dim + 1], m_target_points[rows[i] * dim + dim]);
            }
        }
        for (int j = 0; j < N; j++) {
            for (int dim = 0; dim < dimension; dim++) {
                source_box[2 * dim + 0] = std::max(source_box[2 * dim + 0], m_source_points[cols[j] * dim + dim]);
                source_box[2 * dim + 1] = std::min(source_box[2 * dim + 1], m_source_points[cols[j] * dim + dim]);
            }
        }

        return true;
    }
    // // C style
    // bool copy_low_rank_approximation(int M, int N, const int *rows, const int *cols, LowRankMatrix<CoefficientPrecision> &lrmat) const {
    //     std::array<CoordinatePrecision, 2 * dimension> target_box;
    //     std::array<CoordinatePrecision, 2 * dimension> source_box;
    //     for (int dim = 0; dim < dimension; dim++) {
    //         target_box[2 * dim + 0] = std::numeric_limits<double>::min();
    //         target_box[2 * dim + 1] = std::numeric_limits<double>::max();
    //         source_box[2 * dim + 0] = std::numeric_limits<double>::min();
    //         source_box[2 * dim + 1] = std::numeric_limits<double>::max();
    //     }
    //     for (int target_index : std::span<int>(rows, N)) {
    //         auto &target_elts = m_target_dof_to_elts[target_index];
    //         for (auto [_, elt] : target_elts) {
    //             for (int dim = 0; dim < dimension; dim++) {
    //                 target_box[2 * dim + 0] = std::max(target_box[2 * dim + 0], elt[dim]);
    //                 target_box[2 * dim + 1] = std::min(target_box[2 * dim + 1], elt[dim]);
    //             }
    //         }
    //     }
    //     for (int source_index : std::span<int>(cols, N)) {
    //         auto &source_elts = m_source_dof_to_elts[source_index];
    //         for (auto [_, elt] : source_elts) {
    //             for (int dim = 0; dim < dimension; dim++) {
    //                 source_box[2 * dim + 0] = std::max(source_box[2 * dim + 0], elt[dim]);
    //                 source_box[2 * dim + 1] = std::min(source_box[2 * dim + 1], elt[dim]);
    //             }
    //         }
    //     }
    // }

    bool copy_low_rank_approximation(int M, int N, const int *rows, const int *cols, int reqrank, LowRankMatrix<CoefficientPrecision> &lrmat) const {}
};

} // namespace htool

#endif
