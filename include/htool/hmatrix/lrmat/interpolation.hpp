#ifndef HTOOL_VIRTUAL_IBCOMP_HPP
#define HTOOL_VIRTUAL_IBCOMP_HPP

#include "../../hmatrix/interfaces/virtual_lrmat_generator.hpp" // for Virt...
#include "../../wrappers/wrapper_blas.hpp"                      // for Blas
#include "../../wrappers/wrapper_lapack.hpp"                    // for Blas
#include "general_intrp.hpp"
#include "htool/matrix/utils/output.hpp"
#include "lrmat.hpp"
#include <functional>
#include <iostream>
#include <map>
#include <span>
#include <theia.hpp>

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
    typedef const std::function<void(std::array<CoordinatePrecision, dimension> *, int, std::array<CoordinatePrecision, dimension> *, int, CoefficientPrecision *)> kernel_type;
    kernel_type m_kernel;
    const int *m_target_permutation;
    const int *m_source_permutation;
    const CoordinatePrecision *m_target_points;
    int m_target_size;
    const CoordinatePrecision *m_source_points;
    int m_source_size;
    // const std::map<int, std::vector<std::pair<int, CoordinatePrecision>>> &
    //     m_target_dof_to_elts;
    // const std::map<int, std::vector<std::pair<int, CoordinatePrecision>>> &m_source_dof_to_elts;

  public:
    // HCA(std::function<CoefficientPrecision(std::array<CoordinatePrecision, 3>, std::array<CoordinatePrecision, 3>)> kernel, const int *target_permutation, const std::map<int, std::vector<std::pair<int, CoordinatePrecision>>> &target_dof_to_elts, const int *source_permutation, const std::map<int, std::vector<std::pair<int, CoordinatePrecision>>> &source_dof_to_elts) : m_kernel(kernel), m_target_permutation(target_permutation), m_target_dof_to_elts(target_dof_to_elts), m_source_permutation(source_permutation), m_source_dof_to_elts(source_dof_to_elts) {}

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
            min_x[dim] = std::numeric_limits<double>::max();
        }
        for (int i = 0; i < M; i++) {
            std::array<CoordinatePrecision, dimension> tmp_target_point;
            for (int dim = 0; dim < dimension; dim++) {
                max_x[dim]            = std::max(max_x[dim], m_target_points[rows[i] * dimension + dim]);
                min_x[dim]            = std::min(min_x[dim], m_target_points[rows[i] * dimension + dim]);
                tmp_target_point[dim] = m_target_points[rows[i] * dim + dim];
            }
            local_target_points[i] = tmp_target_point;
        }
        for (int j = 0; j < N; j++) {
            std::array<CoordinatePrecision, dimension> tmp_source_point;
            for (int dim = 0; dim < dimension; dim++) {
                max_y[dim]            = std::max(max_x[dim], m_source_points[cols[j] * dimension + dim]);
                min_y[dim]            = std::min(min_x[dim], m_source_points[cols[j] * dimension + dim]);
                tmp_source_point[dim] = m_source_points[cols[j] * dim + dim];
            }
            local_source_points[j] = tmp_source_point;
        }

        int L = std::ceil(std::log(1. / lrmat.get_epsilon()) / std::log(10));
        // std::cout << lrmat.get_epsilon() << " " << L << " " << min_x[0] << " " << max_x[0] << " " << min_y[0] << " " << max_y[0] << " " << local_target_points[0][0] << " " << local_target_points[0][1] << " " << local_target_points[0][2] << "\n";
        // for (auto &point : local_target_points) {
        //     std::cout << point[0] << " " << point[1] << " " << point[2] << "\n";
        // }
        // theia::lits<CoefficientPrecision, CoordinatePrecision, dimension, kernel_type, 0> general_interpolation(
        //     min_x.data(), max_x.data(), local_target_points.data(), M, min_y.data(), max_y.data(), local_source_points.data(), N, L, &m_kernel);
        // general_interpolation.get_UV(lrmat.get_epsilon());

        auto &U = lrmat.get_U();
        auto &V = lrmat.get_V();
        int rank;
        CoefficientPrecision *U_ptr, *V_ptr;
        // std::cout << "ici\n";
        theia::get_lits_cheb<CoefficientPrecision, CoordinatePrecision, dimension, kernel_type>(min_x.data(), max_x.data(), local_target_points.data(), M, min_y.data(), max_y.data(), local_source_points.data(), N, L, &m_kernel, lrmat.get_epsilon(), U_ptr, V_ptr, rank);
        U.assign(M, rank, U_ptr, true);
        V.assign(rank, N, V_ptr, true);
        // print(U, std::cout, ",");
        // print(V, std::cout, ",");
        // U.resize(M, rank);
        // V.resize(rank, N);

        // for (int i = 0; i < M; i++) {
        //     for (int j = 0; j < rank; j++) {
        //         // std::cout << i << " " << j << " " << general_interpolation.SlU[i + j * M] << "\n";
        //         U(i, j) = general_interpolation.SlU[i + j * M];
        //     }
        // }
        // for (int i = 0; i < N; i++) {
        //     for (int j = 0; j < rank; j++) {
        //         std::cout << general_interpolation.VSr[j + i * rank] << "\n";
        //         V(i, j) = general_interpolation.VSr[j + i * rank];
        //     }
        // }

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
            min_x[dim] = std::numeric_limits<double>::max();
        }
        for (int i = 0; i < M; i++) {
            std::array<CoordinatePrecision, dimension> tmp_target_point;
            for (int dim = 0; dim < dimension; dim++) {
                max_x[dim]            = std::max(max_x[dim], m_target_points[rows[i] * dimension + dim]);
                min_x[dim]            = std::min(min_x[dim], m_target_points[rows[i] * dimension + dim]);
                tmp_target_point[dim] = m_target_points[rows[i] * dim + dim];
            }
            local_target_points[i] = tmp_target_point;
        }
        for (int j = 0; j < N; j++) {
            std::array<CoordinatePrecision, dimension> tmp_source_point;
            for (int dim = 0; dim < dimension; dim++) {
                max_y[dim]            = std::max(max_x[dim], m_source_points[cols[j] * dimension + dim]);
                min_y[dim]            = std::min(min_x[dim], m_source_points[cols[j] * dimension + dim]);
                tmp_source_point[dim] = m_source_points[cols[j] * dim + dim];
            }
            local_target_points[j] = tmp_source_point;
        }

        int L = std::ceil(log(1. / lrmat.get_epsilon()));
        // theia::lits<CoefficientPrecision, CoordinatePrecision, dimension, kernel_type, 0> general_interpolation(
        //     min_x.data(), max_x.data(), local_target_points.data(), M, min_y.data(), max_y.data(), local_source_points.data(), N, L, &m_kernel);
        // general_interpolation.get_UV(lrmat.get_epsilon());

        // auto &U = lrmat.get_U();
        // auto &V = lrmat.get_V();
        // if (general_interpolation.UV.r < reqrank) {
        //     std::cout << general_interpolation.UV.r << "\n";
        //     exit(1);
        // }
        // int rank = general_interpolation.UV.r;
        // U.resize(M, rank);
        // V.resize(rank, N);
        // for (int i = 0; i < M; i++) {
        //     for (int j = 0; j < rank; j++) {
        //         U(i, j) = general_interpolation.SlU[i + j * M];
        //     }
        // }
        // for (int i = 0; i < N; i++) {
        //     for (int j = 0; j < rank; j++) {
        //         V(i, j) = general_interpolation.VSr[i + j * N];
        //     }
        // }
        auto &U  = lrmat.get_U();
        auto &V  = lrmat.get_V();
        int rank = reqrank;
        CoefficientPrecision *U_ptr, *V_ptr;
        // std::cout << "ici\n";
        theia::get_lits_cheb_fixed_rank<CoefficientPrecision, CoordinatePrecision, dimension, kernel_type>(min_x.data(), max_x.data(), local_target_points.data(), M, min_y.data(), max_y.data(), local_source_points.data(), N, L, &m_kernel, lrmat.get_epsilon(), U_ptr, V_ptr, rank);
        // std::cout << "la\n";
        U.assign(M, rank, U_ptr, true);
        V.assign(rank, N, V_ptr, true);
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
};

} // namespace htool

#endif
