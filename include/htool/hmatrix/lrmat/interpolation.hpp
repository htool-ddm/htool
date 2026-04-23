#ifndef HTOOL_VIRTUAL_IBCOMP_HPP
#define HTOOL_VIRTUAL_IBCOMP_HPP

#include "../../hmatrix/interfaces/virtual_lrmat_generator.hpp" // for Virt...
#include <functional>
#include <iostream>
#include <map>
#include <span>

#if defined(__clang__)
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wold-style-cast"
#    pragma clang diagnostic ignored "-Wdouble-promotion"
#    pragma clang diagnostic ignored "-Wunused-parameter"
#    pragma clang diagnostic ignored "-Wshadow"
#elif defined(__GNUC__) || defined(__GNUG__)
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wdouble-promotion"
#    pragma GCC diagnostic ignored "-Wmismatched-new-delete"
#    pragma GCC diagnostic ignored "-Wold-style-cast"
#    pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

#include <theia.hpp>

#if defined(__clang__)
#    pragma clang diagnostic pop
#elif defined(__GNUC__) || defined(__GNUG__)
#    pragma GCC diagnostic pop
#endif

namespace htool {

template <typename CoefficientPrecision, typename CoordinatePrecision, int dimension>
class HCA final : public VirtualInternalLowRankGenerator<CoefficientPrecision> {

  private:
    typedef const std::function<void(std::array<CoordinatePrecision, dimension> *, int, std::array<CoordinatePrecision, dimension> *, int, CoefficientPrecision *)> kernel_type;
    kernel_type m_kernel;
    const CoordinatePrecision *m_target_points;
    int m_target_size;
    const int *m_target_permutation;
    const CoordinatePrecision *m_source_points;
    int m_source_size;
    const int *m_source_permutation;

  public:
    HCA(kernel_type kernel, CoordinatePrecision *target_points, int target_size, const int *target_permutation, CoordinatePrecision *source_points, int source_size, const int *source_permutation) : m_kernel(kernel), m_target_points(target_points), m_target_size(target_size), m_target_permutation(target_permutation), m_source_points(source_points), m_source_size(source_size), m_source_permutation(source_permutation) {}

    virtual bool copy_low_rank_approximation(int M, int N, int row_offset, int col_offset, LowRankMatrix<CoefficientPrecision> &lrmat) const override {
        return copy_low_rank_approximation(M, N, m_target_permutation + row_offset, m_source_permutation + col_offset, lrmat);
    }

    virtual bool copy_low_rank_approximation(int M, int N, int row_offset, int col_offset, int reqrank, LowRankMatrix<CoefficientPrecision> &lrmat) const override {
        return copy_low_rank_approximation(M, N, m_target_permutation + row_offset, m_source_permutation + col_offset, reqrank, lrmat);
    }

    bool copy_low_rank_approximation(int M, int N, const int *rows, const int *cols, LowRankMatrix<CoefficientPrecision> &lrmat) const {
        std::array<CoordinatePrecision, dimension> max_x, max_y;
        std::array<CoordinatePrecision, dimension> min_x, min_y;
        std::vector<std::array<CoordinatePrecision, dimension>> local_target_points(M);
        std::vector<std::array<CoordinatePrecision, dimension>> local_source_points(N);

        for (int dim = 0; dim < dimension; dim++) {
            max_x[dim] = std::numeric_limits<double>::min();
            min_x[dim] = std::numeric_limits<double>::max();
            max_y[dim] = std::numeric_limits<double>::min();
            min_y[dim] = std::numeric_limits<double>::max();
        }
        for (int i = 0; i < M; i++) {
            std::array<CoordinatePrecision, dimension> tmp_target_point;
            for (int dim = 0; dim < dimension; dim++) {
                max_x[dim]            = std::max(max_x[dim], m_target_points[rows[i] * dimension + dim]);
                min_x[dim]            = std::min(min_x[dim], m_target_points[rows[i] * dimension + dim]);
                tmp_target_point[dim] = m_target_points[rows[i] * dimension + dim];
            }
            local_target_points[i] = tmp_target_point;
        }
        for (int j = 0; j < N; j++) {
            std::array<CoordinatePrecision, dimension> tmp_source_point;
            for (int dim = 0; dim < dimension; dim++) {
                max_y[dim]            = std::max(max_y[dim], m_source_points[cols[j] * dimension + dim]);
                min_y[dim]            = std::min(min_y[dim], m_source_points[cols[j] * dimension + dim]);
                tmp_source_point[dim] = m_source_points[cols[j] * dimension + dim];
            }
            local_source_points[j] = tmp_source_point;
        }

        int L   = std::ceil(std::log(1. / lrmat.get_epsilon()) / std::log(10));
        auto &U = lrmat.get_U();
        auto &V = lrmat.get_V();
        int rank;
        CoefficientPrecision *U_ptr, *V_ptr;
        theia::get_lits_cheb<CoefficientPrecision, CoordinatePrecision, dimension, kernel_type>(min_x.data(), max_x.data(), local_target_points.data(), M, min_y.data(), max_y.data(), local_source_points.data(), N, L, &m_kernel, lrmat.get_epsilon(), U_ptr, V_ptr, rank);
        U.assign(M, rank, U_ptr, true);
        V.assign(rank, N, V_ptr, true);

        return true;
    }

    bool copy_low_rank_approximation(int M, int N, const int *rows, const int *cols, int reqrank, LowRankMatrix<CoefficientPrecision> &lrmat) const {

        std::array<CoordinatePrecision, dimension> max_x, max_y;
        std::array<CoordinatePrecision, dimension> min_x, min_y;
        std::vector<std::array<CoordinatePrecision, dimension>> local_target_points(M);
        std::vector<std::array<CoordinatePrecision, dimension>> local_source_points(N);

        for (int dim = 0; dim < dimension; dim++) {
            max_x[dim] = std::numeric_limits<double>::min();
            min_x[dim] = std::numeric_limits<double>::max();
            max_y[dim] = std::numeric_limits<double>::min();
            min_y[dim] = std::numeric_limits<double>::max();
        }
        for (int i = 0; i < M; i++) {
            std::array<CoordinatePrecision, dimension> tmp_target_point;
            for (int dim = 0; dim < dimension; dim++) {
                max_x[dim]            = std::max(max_x[dim], m_target_points[rows[i] * dimension + dim]);
                min_x[dim]            = std::min(min_x[dim], m_target_points[rows[i] * dimension + dim]);
                tmp_target_point[dim] = m_target_points[rows[i] * dimension + dim];
            }
            local_target_points[i] = tmp_target_point;
        }
        for (int j = 0; j < N; j++) {
            std::array<CoordinatePrecision, dimension> tmp_source_point;
            for (int dim = 0; dim < dimension; dim++) {
                max_y[dim]            = std::max(max_y[dim], m_source_points[cols[j] * dimension + dim]);
                min_y[dim]            = std::min(min_y[dim], m_source_points[cols[j] * dimension + dim]);
                tmp_source_point[dim] = m_source_points[cols[j] * dimension + dim];
            }
            local_source_points[j] = tmp_source_point;
        }

        int L    = std::ceil(std::log(1. / lrmat.get_epsilon()) / std::log(10));
        auto &U  = lrmat.get_U();
        auto &V  = lrmat.get_V();
        int rank = reqrank;
        CoefficientPrecision *U_ptr, *V_ptr;
        theia::get_lits_cheb_fixed_rank<CoefficientPrecision, CoordinatePrecision, dimension, kernel_type>(min_x.data(), max_x.data(), local_target_points.data(), M, min_y.data(), max_y.data(), local_source_points.data(), N, L, &m_kernel, lrmat.get_epsilon(), U_ptr, V_ptr, rank);
        U.assign(M, rank, U_ptr, true);
        V.assign(rank, N, V_ptr, true);
        return true;
    }
};

// template <typename CoefficientPrecision, typename CoordinatePrecision, int dimension>
// class BEMHCA final : public VirtualInternalLowRankGenerator<CoefficientPrecision> {

//   private:
//     typedef const std::function<void(std::array<CoordinatePrecision, dimension> *, int, std::array<CoordinatePrecision, dimension> *, int, CoefficientPrecision *)> kernel_type;
//     kernel_type m_kernel;
//     const int *m_target_permutation;
//     const int *m_source_permutation;
//     const CoordinatePrecision *m_target_points;
//     int m_target_size;
//     const CoordinatePrecision *m_source_points;
//     int m_source_size;
//     // const std::map<int, std::vector<std::pair<int, CoordinatePrecision>>> &
//     //     m_target_dof_to_elts;
//     // const std::map<int, std::vector<std::pair<int, CoordinatePrecision>>> &m_source_dof_to_elts;

//   public:
//     HCA(kernel_type kernel, std::map<int, std::vector<int>> target_dofs_to_elements, int *target_elements_to_points, int target_number_of_elements, CoordinatePrecision *target_points, int target_size, const int *source_permutation, std::map<int, std::vector<int>> source_dofs_to_elements, int *source_elements_to_points, int source_number_of_elements, CoordinatePrecision *source_points, int source_size, const int *source_permutation) : m_kernel(kernel), m_target_points(target_points), m_target_size(target_size), m_target_permutation(target_permutation), m_source_points(source_points), m_source_size(source_size), m_source_permutation(source_permutation) {}

//     virtual bool copy_low_rank_approximation(int M, int N, int row_offset, int col_offset, LowRankMatrix<CoefficientPrecision> &lrmat) const override {
//         return copy_low_rank_approximation(M, N, m_target_permutation + row_offset, m_source_permutation + col_offset, lrmat);
//     }

//     virtual bool copy_low_rank_approximation(int M, int N, int row_offset, int col_offset, int reqrank, LowRankMatrix<CoefficientPrecision> &lrmat) const override {
//         return copy_low_rank_approximation(M, N, m_target_permutation + row_offset, m_source_permutation + col_offset, reqrank, lrmat);
//     }

//     bool copy_low_rank_approximation(int M, int N, const int *rows, const int *cols, LowRankMatrix<CoefficientPrecision> &lrmat) const {
//         // std::array<CoordinatePrecision, dimension> max_x, max_y;
//         // std::array<CoordinatePrecision, dimension> min_x, min_y;
//         // std::vector<std::array<CoordinatePrecision, dimension>> local_target_points(M);
//         // std::vector<std::array<CoordinatePrecision, dimension>> local_source_points(N);

//         // for (int dim = 0; dim < dimension; dim++) {
//         //     max_x[dim] = std::numeric_limits<double>::min();
//         //     min_x[dim] = std::numeric_limits<double>::max();
//         //     max_y[dim] = std::numeric_limits<double>::min();
//         //     min_y[dim] = std::numeric_limits<double>::max();
//         // }
//         // for (int i = 0; i < M; i++) {
//         //     std::array<CoordinatePrecision, dimension> tmp_target_point;
//         //     for (int dim = 0; dim < dimension; dim++) {
//         //         max_x[dim]            = std::max(max_x[dim], m_target_points[rows[i] * dimension + dim]);
//         //         min_x[dim]            = std::min(min_x[dim], m_target_points[rows[i] * dimension + dim]);
//         //         tmp_target_point[dim] = m_target_points[rows[i] * dimension + dim];
//         //     }
//         //     local_target_points[i] = tmp_target_point;
//         // }
//         // for (int j = 0; j < N; j++) {
//         //     std::array<CoordinatePrecision, dimension> tmp_source_point;
//         //     for (int dim = 0; dim < dimension; dim++) {
//         //         max_y[dim]            = std::max(max_y[dim], m_source_points[cols[j] * dimension + dim]);
//         //         min_y[dim]            = std::min(min_y[dim], m_source_points[cols[j] * dimension + dim]);
//         //         tmp_source_point[dim] = m_source_points[cols[j] * dimension + dim];
//         //     }
//         //     local_source_points[j] = tmp_source_point;
//         // }

//         // int L   = std::ceil(std::log(1. / lrmat.get_epsilon()) / std::log(10)) + 1;
//         // auto &U = lrmat.get_U();
//         // auto &V = lrmat.get_V();
//         // int rank;
//         // CoefficientPrecision *U_ptr, *V_ptr;
//         // theia::get_lits_cheb<CoefficientPrecision, CoordinatePrecision, dimension, kernel_type>(min_x.data(), max_x.data(), local_target_points.data(), M, min_y.data(), max_y.data(), local_source_points.data(), N, L, &m_kernel, lrmat.get_epsilon(), U_ptr, V_ptr, rank);
//         // U.assign(M, rank, U_ptr, true);
//         // V.assign(rank, N, V_ptr, true);

//         // return true;
//     }

//     bool copy_low_rank_approximation(int M, int N, const int *rows, const int *cols, int reqrank, LowRankMatrix<CoefficientPrecision> &lrmat) const {

//         // std::array<CoordinatePrecision, dimension> max_x, max_y;
//         // std::array<CoordinatePrecision, dimension> min_x, min_y;
//         // std::vector<std::array<CoordinatePrecision, dimension>> local_target_points(M);
//         // std::vector<std::array<CoordinatePrecision, dimension>> local_source_points(N);

//         // for (int dim = 0; dim < dimension; dim++) {
//         //     max_x[dim] = std::numeric_limits<double>::min();
//         //     min_x[dim] = std::numeric_limits<double>::max();
//         //     max_y[dim] = std::numeric_limits<double>::min();
//         //     min_y[dim] = std::numeric_limits<double>::max();
//         // }
//         // for (int i = 0; i < M; i++) {
//         //     std::array<CoordinatePrecision, dimension> tmp_target_point;
//         //     for (int dim = 0; dim < dimension; dim++) {
//         //         max_x[dim]            = std::max(max_x[dim], m_target_points[rows[i] * dimension + dim]);
//         //         min_x[dim]            = std::min(min_x[dim], m_target_points[rows[i] * dimension + dim]);
//         //         tmp_target_point[dim] = m_target_points[rows[i] * dimension + dim];
//         //     }
//         //     local_target_points[i] = tmp_target_point;
//         // }
//         // for (int j = 0; j < N; j++) {
//         //     std::array<CoordinatePrecision, dimension> tmp_source_point;
//         //     for (int dim = 0; dim < dimension; dim++) {
//         //         max_y[dim]            = std::max(max_y[dim], m_source_points[cols[j] * dimension + dim]);
//         //         min_y[dim]            = std::min(min_y[dim], m_source_points[cols[j] * dimension + dim]);
//         //         tmp_source_point[dim] = m_source_points[cols[j] * dimension + dim];
//         //     }
//         //     local_source_points[j] = tmp_source_point;
//         // }

//         // int L    = std::ceil(std::log(1. / lrmat.get_epsilon()) / std::log(10));
//         // auto &U  = lrmat.get_U();
//         // auto &V  = lrmat.get_V();
//         // int rank = reqrank;
//         // CoefficientPrecision *U_ptr, *V_ptr;
//         // theia::get_lits_cheb_fixed_rank<CoefficientPrecision, CoordinatePrecision, dimension, kernel_type>(min_x.data(), max_x.data(), local_target_points.data(), M, min_y.data(), max_y.data(), local_source_points.data(), N, L, &m_kernel, lrmat.get_epsilon(), U_ptr, V_ptr, rank);
//         // U.assign(M, rank, U_ptr, true);
//         // V.assign(rank, N, V_ptr, true);
//         // // std::cout << V.nb_rows() << " " << V.nb_cols() << "\n";
//         // // print(V, std::cout, ",");
//         // return true;
//     }
// };

} // namespace htool

#endif
