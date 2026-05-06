#ifndef HTOOL_VIRTUAL_IBCOMP_HPP
#define HTOOL_VIRTUAL_IBCOMP_HPP

#include "../../hmatrix/interfaces/virtual_lrmat_generator.hpp" // for Virt...
#include "../../quadratures/geometry.hpp"
#include "htool/quadratures/triangle.hpp"
#include <functional>
#include <map>
#include <set>
#include <utility>

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

template <typename CoefficientPrecision, typename CoordinatePrecision, int dimension>
class BEMHCA final : public VirtualInternalLowRankGenerator<CoefficientPrecision> {

  private:
    typedef const std::function<void(std::array<CoordinatePrecision, dimension> *, int, std::array<CoordinatePrecision, dimension> *, int, CoefficientPrecision *)> kernel_type;
    kernel_type m_kernel;
    typedef const std::function<CoefficientPrecision(const std::vector<std::array<CoordinatePrecision, dimension>> &, int, const std::array<CoordinatePrecision, dimension> &)> basis_function_type;
    basis_function_type m_target_basis_function;
    std::map<int, std::vector<int>> m_target_dofs_to_elements;
    const int *m_target_elements_to_points;
    int m_target_number_of_points_per_element;
    const CoordinatePrecision *m_target_points;
    int m_target_size;
    const int *m_target_permutation;
    basis_function_type m_source_basis_function;
    std::map<int, std::vector<int>> m_source_dofs_to_elements;
    const int *m_source_elements_to_points;
    int m_source_number_of_points_per_element;
    const CoordinatePrecision *m_source_points;
    int m_source_size;
    const int *m_source_permutation;

  public:
    BEMHCA(kernel_type kernel, basis_function_type target_basis_function, std::map<int, std::vector<int>> target_dofs_to_elements, const int *target_elements_to_points, int target_number_of_points_per_element, CoordinatePrecision *target_points, int target_size, const int *target_permutation, basis_function_type source_basis_function, std::map<int, std::vector<int>> source_dofs_to_elements, int *source_elements_to_points, int source_number_of_points_per_element, CoordinatePrecision *source_points, int source_size, const int *source_permutation) : m_kernel(kernel), m_target_basis_function(target_basis_function), m_target_dofs_to_elements(target_dofs_to_elements), m_target_elements_to_points(target_elements_to_points), m_target_number_of_points_per_element(target_number_of_points_per_element), m_target_points(target_points), m_target_size(target_size), m_target_permutation(target_permutation), m_source_basis_function(source_basis_function), m_source_dofs_to_elements(source_dofs_to_elements), m_source_elements_to_points(source_elements_to_points), m_source_number_of_points_per_element(source_number_of_points_per_element), m_source_points(source_points), m_source_size(source_size), m_source_permutation(source_permutation) {}

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
        std::set<int> target_element_indices;
        std::set<int> source_element_indices;
        std::map<std::pair<int, int>, int> target_local_dof_index_to_global;
        std::map<std::pair<int, int>, int> source_local_dof_index_to_global;
        int L = std::ceil(std::log(1. / lrmat.get_epsilon()) / std::log(10)) + 1;
        Matrix<CoefficientPrecision> U_tilde(M, std::pow(L, dimension));
        Matrix<CoefficientPrecision> V_tilde(std::pow(L, dimension), N);

        for (int dim = 0; dim < dimension; dim++) {
            max_x[dim] = std::numeric_limits<double>::min();
            min_x[dim] = std::numeric_limits<double>::max();
            max_y[dim] = std::numeric_limits<double>::min();
            min_y[dim] = std::numeric_limits<double>::max();
        }
        for (int i = 0; i < M; i++) {
            const std::vector<int> &target_elt_indices = m_target_dofs_to_elements.at(rows[i]);
            for (auto elt_index : target_elt_indices) {
                target_element_indices.insert(elt_index);
                for (int p = 0; p < m_target_number_of_points_per_element; p++) {
                    target_local_dof_index_to_global[std::make_pair(elt_index, p)] = i;

                    int point_index = m_target_elements_to_points[elt_index * m_target_number_of_points_per_element + p];
                    for (int dim = 0; dim < dimension; dim++) {
                        max_x[dim] = std::max(max_x[dim], m_target_points[point_index * dimension + dim]);
                        min_x[dim] = std::min(min_x[dim], m_target_points[point_index * dimension + dim]);
                    }
                }
            }
        }
        for (int j = 0; j < N; j++) {
            const std::vector<int> &source_elt_indices = m_source_dofs_to_elements.at(cols[j]);
            for (auto elt_index : source_elt_indices) {
                source_element_indices.insert(elt_index);
                for (int p = 0; p < m_source_number_of_points_per_element; p++) {
                    source_local_dof_index_to_global[std::make_pair(elt_index, p)] = j;

                    int point_index = m_source_elements_to_points[elt_index * m_source_number_of_points_per_element + p];
                    for (int dim = 0; dim < dimension; dim++) {
                        max_x[dim] = std::max(max_x[dim], m_source_points[point_index * dimension + dim]);
                        min_x[dim] = std::min(min_x[dim], m_source_points[point_index * dimension + dim]);
                    }
                }
            }
        }

        std::vector<std::array<double, dimension>> local_target_quadrature_points;
        std::vector<std::array<double, dimension>> tmp_cell_points(m_target_number_of_points_per_element);
        for (auto target_element_index : target_element_indices) {
            for (int p = 0; p < m_target_number_of_points_per_element; p++) {
                int point_index = m_target_elements_to_points[target_element_index * m_target_number_of_points_per_element + p];
                for (int dim = 0; dim < dimension; dim++) {
                    tmp_cell_points[p][dim] = m_target_points[point_index * dimension + dim];
                }
            }
            CoordinatePrecision jacobian = 0;
            std::vector<CoefficientPrecision> weights(triangle_rules<CoordinatePrecision>[10].nb_points);
            Matrix<CoefficientPrecision> tmp_mat(std::pow(L, dimension), m_target_number_of_points_per_element);
            for (int i = 0; i < triangle_rules<CoordinatePrecision>[10].nb_points; i++) {
                weights[i] = triangle_rules<CoordinatePrecision>[10].quad_points[i].w;
            }
            if (m_target_number_of_points_per_element == 3) {
                jacobian                       = triangle_jacobian(tmp_cell_points[0], tmp_cell_points[1], tmp_cell_points[2]);
                local_target_quadrature_points = map_dunavant_to_triangle(tmp_cell_points[0], tmp_cell_points[1], tmp_cell_points[2], triangle_rules<CoordinatePrecision>[10]);
                CoefficientPrecision *S_data;
                theia::get_polynomials<dimension, CoordinatePrecision, CoefficientPrecision, 0>(L, S_data, min_x.data(), max_x.data(), local_target_quadrature_points.data(), local_target_quadrature_points.size());
                Matrix<CoefficientPrecision> S;
                S.assign(std::pow(L, dimension), local_target_quadrature_points.size(), S_data, true);
                Matrix<CoefficientPrecision> phi(local_target_quadrature_points.size(), m_target_number_of_points_per_element);
                for (int i = 0; i < phi.nb_rows(); i++) {
                    for (int j = 0; j < phi.nb_cols(); j++) {
                        phi(i, j) = jacobian * weights[i] * m_target_basis_function(tmp_cell_points, j, local_target_quadrature_points[i]);
                    }
                }
                add_matrix_matrix_product('N', 'N', CoefficientPrecision(1), S, phi, CoefficientPrecision(0), tmp_mat);
                for (int i = 0; i < tmp_mat.nb_rows(); i++) {
                    for (int j = 0; j < m_target_number_of_points_per_element; j++) {
                        U_tilde(target_local_dof_index_to_global[{target_element_index, j}], i) += tmp_mat(i, j);
                    }
                }
            }
        }

        // meme chose en source

        // auto &U = lrmat.get_U();
        // auto &V = lrmat.get_V();
        // int rank;
        // CoefficientPrecision *U_ptr, *V_ptr;
        // theia::get_lits_cheb<CoefficientPrecision, CoordinatePrecision, dimension, kernel_type>(min_x.data(), max_x.data(), local_target_points.data(), M, min_y.data(), max_y.data(), local_source_points.data(), N, L, &m_kernel, lrmat.get_epsilon(), U_ptr, V_ptr, rank);
        // U.assign(M, rank, U_ptr, true);
        // V.assign(rank, N, V_ptr, true);

        return true;
    }

    bool copy_low_rank_approximation(int M, int N, const int *rows, const int *cols, int reqrank, LowRankMatrix<CoefficientPrecision> &lrmat) const {

        // std::array<CoordinatePrecision, dimension> max_x, max_y;
        // std::array<CoordinatePrecision, dimension> min_x, min_y;
        // std::vector<std::array<CoordinatePrecision, dimension>> local_target_points(M);
        // std::vector<std::array<CoordinatePrecision, dimension>> local_source_points(N);

        // for (int dim = 0; dim < dimension; dim++) {
        //     max_x[dim] = std::numeric_limits<double>::min();
        //     min_x[dim] = std::numeric_limits<double>::max();
        //     max_y[dim] = std::numeric_limits<double>::min();
        //     min_y[dim] = std::numeric_limits<double>::max();
        // }
        // for (int i = 0; i < M; i++) {
        //     std::array<CoordinatePrecision, dimension> tmp_target_point;
        //     for (int dim = 0; dim < dimension; dim++) {
        //         max_x[dim]            = std::max(max_x[dim], m_target_points[rows[i] * dimension + dim]);
        //         min_x[dim]            = std::min(min_x[dim], m_target_points[rows[i] * dimension + dim]);
        //         tmp_target_point[dim] = m_target_points[rows[i] * dimension + dim];
        //     }
        //     local_target_points[i] = tmp_target_point;
        // }
        // for (int j = 0; j < N; j++) {
        //     std::array<CoordinatePrecision, dimension> tmp_source_point;
        //     for (int dim = 0; dim < dimension; dim++) {
        //         max_y[dim]            = std::max(max_y[dim], m_source_points[cols[j] * dimension + dim]);
        //         min_y[dim]            = std::min(min_y[dim], m_source_points[cols[j] * dimension + dim]);
        //         tmp_source_point[dim] = m_source_points[cols[j] * dimension + dim];
        //     }
        //     local_source_points[j] = tmp_source_point;
        // }

        // int L    = std::ceil(std::log(1. / lrmat.get_epsilon()) / std::log(10));
        // auto &U  = lrmat.get_U();
        // auto &V  = lrmat.get_V();
        // int rank = reqrank;
        // CoefficientPrecision *U_ptr, *V_ptr;
        // theia::get_lits_cheb_fixed_rank<CoefficientPrecision, CoordinatePrecision, dimension, kernel_type>(min_x.data(), max_x.data(), local_target_points.data(), M, min_y.data(), max_y.data(), local_source_points.data(), N, L, &m_kernel, lrmat.get_epsilon(), U_ptr, V_ptr, rank);
        // U.assign(M, rank, U_ptr, true);
        // V.assign(rank, N, V_ptr, true);
        // // std::cout << V.nb_rows() << " " << V.nb_cols() << "\n";
        // // print(V, std::cout, ",");
        // return true;
    }
};

} // namespace htool

#endif
