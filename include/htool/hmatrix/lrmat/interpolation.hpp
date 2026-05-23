#ifndef HTOOL_VIRTUAL_IBCOMP_HPP
#define HTOOL_VIRTUAL_IBCOMP_HPP

#include "../../hmatrix/interfaces/virtual_lrmat_generator.hpp" // for Virt...
#include "../../quadratures/geometry.hpp"
#include "htool/matrix/linalg/add_matrix_matrix_product.hpp"
#include "htool/matrix/utils/SVD_truncation.hpp"
#include "htool/misc/misc.hpp"
#include "htool/quadratures/gauss_legendre.hpp"
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
  public:
    typedef const std::function<void(std::array<CoordinatePrecision, dimension> *, int, std::array<CoordinatePrecision, dimension> *, int, CoefficientPrecision *)> kernel_type;

  private:
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
        std::array<CoordinatePrecision, dimension> max_target_box, max_source_box;
        std::array<CoordinatePrecision, dimension> min_target_box, min_source_box;
        std::vector<std::array<CoordinatePrecision, dimension>> local_target_points(M);
        std::vector<std::array<CoordinatePrecision, dimension>> local_source_points(N);

        for (int dim = 0; dim < dimension; dim++) {
            max_target_box[dim] = std::numeric_limits<double>::min();
            min_target_box[dim] = std::numeric_limits<double>::max();
            max_source_box[dim] = std::numeric_limits<double>::min();
            min_source_box[dim] = std::numeric_limits<double>::max();
        }
        for (int i = 0; i < M; i++) {
            std::array<CoordinatePrecision, dimension> tmp_target_point;
            for (int dim = 0; dim < dimension; dim++) {
                max_target_box[dim]   = std::max(max_target_box[dim], m_target_points[rows[i] * dimension + dim]);
                min_target_box[dim]   = std::min(min_target_box[dim], m_target_points[rows[i] * dimension + dim]);
                tmp_target_point[dim] = m_target_points[rows[i] * dimension + dim];
            }
            local_target_points[i] = tmp_target_point;
        }
        for (int j = 0; j < N; j++) {
            std::array<CoordinatePrecision, dimension> tmp_source_point;
            for (int dim = 0; dim < dimension; dim++) {
                max_source_box[dim]   = std::max(max_source_box[dim], m_source_points[cols[j] * dimension + dim]);
                min_source_box[dim]   = std::min(min_source_box[dim], m_source_points[cols[j] * dimension + dim]);
                tmp_source_point[dim] = m_source_points[cols[j] * dimension + dim];
            }
            local_source_points[j] = tmp_source_point;
        }

        int L   = std::ceil(std::log(1. / lrmat.get_epsilon()) / std::log(10));
        auto &U = lrmat.get_U();
        auto &V = lrmat.get_V();
        int rank;
        CoefficientPrecision *U_ptr, *V_ptr;
        theia::get_lits_cheb<CoefficientPrecision, CoordinatePrecision, dimension, kernel_type>(min_target_box.data(), max_target_box.data(), local_target_points.data(), M, min_source_box.data(), max_source_box.data(), local_source_points.data(), N, L, &m_kernel, lrmat.get_epsilon(), U_ptr, V_ptr, rank);
        U.assign(M, rank, U_ptr, true);
        V.assign(rank, N, V_ptr, true);

        return true;
    }

    bool copy_low_rank_approximation(int M, int N, const int *rows, const int *cols, int reqrank, LowRankMatrix<CoefficientPrecision> &lrmat) const {

        std::array<CoordinatePrecision, dimension> max_target_box, max_source_box;
        std::array<CoordinatePrecision, dimension> min_target_box, min_source_box;
        std::vector<std::array<CoordinatePrecision, dimension>> local_target_points(M);
        std::vector<std::array<CoordinatePrecision, dimension>> local_source_points(N);

        for (int dim = 0; dim < dimension; dim++) {
            max_target_box[dim] = std::numeric_limits<double>::min();
            min_target_box[dim] = std::numeric_limits<double>::max();
            max_source_box[dim] = std::numeric_limits<double>::min();
            min_source_box[dim] = std::numeric_limits<double>::max();
        }
        for (int i = 0; i < M; i++) {
            std::array<CoordinatePrecision, dimension> tmp_target_point;
            for (int dim = 0; dim < dimension; dim++) {
                max_target_box[dim]   = std::max(max_target_box[dim], m_target_points[rows[i] * dimension + dim]);
                min_target_box[dim]   = std::min(min_target_box[dim], m_target_points[rows[i] * dimension + dim]);
                tmp_target_point[dim] = m_target_points[rows[i] * dimension + dim];
            }
            local_target_points[i] = tmp_target_point;
        }
        for (int j = 0; j < N; j++) {
            std::array<CoordinatePrecision, dimension> tmp_source_point;
            for (int dim = 0; dim < dimension; dim++) {
                max_source_box[dim]   = std::max(max_source_box[dim], m_source_points[cols[j] * dimension + dim]);
                min_source_box[dim]   = std::min(min_source_box[dim], m_source_points[cols[j] * dimension + dim]);
                tmp_source_point[dim] = m_source_points[cols[j] * dimension + dim];
            }
            local_source_points[j] = tmp_source_point;
        }

        int L    = std::ceil(std::log(1. / lrmat.get_epsilon()) / std::log(10));
        auto &U  = lrmat.get_U();
        auto &V  = lrmat.get_V();
        int rank = reqrank;
        CoefficientPrecision *U_ptr, *V_ptr;
        theia::get_lits_cheb_fixed_rank<CoefficientPrecision, CoordinatePrecision, dimension, kernel_type>(min_target_box.data(), max_target_box.data(), local_target_points.data(), M, min_source_box.data(), max_source_box.data(), local_source_points.data(), N, L, &m_kernel, lrmat.get_epsilon(), U_ptr, V_ptr, rank);
        U.assign(M, rank, U_ptr, true);
        V.assign(rank, N, V_ptr, true);
        return true;
    }
};

template <typename CoefficientPrecision, typename CoordinatePrecision, int dimension>
class BEMHCA final : public VirtualInternalLowRankGenerator<CoefficientPrecision> {
  public:
    typedef const std::function<void(std::array<CoordinatePrecision, dimension> *, int, std::array<CoordinatePrecision, dimension> *, int, CoefficientPrecision *)> kernel_type;
    typedef const std::function<CoefficientPrecision(const std::vector<std::array<CoordinatePrecision, dimension>> &, int, const std::array<CoordinatePrecision, dimension> &)> basis_function_type;

  private:
    kernel_type m_kernel;
    basis_function_type m_target_basis_function;
    std::map<int, std::vector<int>> m_target_dofs_to_elements;
    int m_target_number_of_dofs_per_element;
    const int *m_target_elements_to_points;
    int m_target_number_of_points_per_element;
    const CoordinatePrecision *m_target_points;
    int m_target_size;
    const int *m_target_permutation;
    int m_target_quadrature_order;
    basis_function_type m_source_basis_function;
    std::map<int, std::vector<int>> m_source_dofs_to_elements;
    int m_source_number_of_dofs_per_element;
    const int *m_source_elements_to_points;
    int m_source_number_of_points_per_element;
    const CoordinatePrecision *m_source_points;
    int m_source_size;
    const int *m_source_permutation;
    int m_source_quadrature_order;

    bool copy_low_rank_approximation(int M, int N, const int *rows, const int *cols, underlying_type<CoefficientPrecision> epsilon, int reqrank, LowRankMatrix<CoefficientPrecision> &lrmat) const {
        std::array<CoordinatePrecision, dimension> max_target_box, max_source_box;
        std::array<CoordinatePrecision, dimension> min_target_box, min_source_box;
        std::vector<std::array<CoordinatePrecision, dimension>> local_target_points(M);
        std::vector<std::array<CoordinatePrecision, dimension>> local_source_points(N);
        std::set<int> target_element_indices;
        std::set<int> source_element_indices;
        std::map<std::pair<int, int>, int> target_element_indices_to_dofs;
        std::map<std::pair<int, int>, int> source_element_indices_to_dofs;
        int L = std::ceil(std::log(1. / epsilon) / std::log(10)) + 1;
        Matrix<CoefficientPrecision> U_tilde(M, std::pow(L, dimension));
        Matrix<CoefficientPrecision> V_tilde(std::pow(L, dimension), N);
        Matrix<CoefficientPrecision> K(std::pow(L, dimension), std::pow(L, dimension));

        // Compute boxes
        for (int dim = 0; dim < dimension; dim++) {
            max_target_box[dim] = std::numeric_limits<double>::min();
            min_target_box[dim] = std::numeric_limits<double>::max();
            max_source_box[dim] = std::numeric_limits<double>::min();
            min_source_box[dim] = std::numeric_limits<double>::max();
        }
        for (int i = 0; i < M; i++) {
            const std::vector<int> &target_elt_indices = m_target_dofs_to_elements.at(rows[i]);
            int number_of_elements_sharing_dof         = target_elt_indices.size() / 2;
            for (int l = 0; l < number_of_elements_sharing_dof; l++) {
                target_element_indices.insert(target_elt_indices[l * 2]);
                target_element_indices_to_dofs[std::make_pair(target_elt_indices[l * 2], target_elt_indices[l * 2 + 1])] = i;

                for (int p = 0; p < m_target_number_of_points_per_element; p++) {

                    int point_index = m_target_elements_to_points[target_elt_indices[l * 2] * m_target_number_of_points_per_element + p];
                    for (int dim = 0; dim < dimension; dim++) {
                        max_target_box[dim] = std::max(max_target_box[dim], m_target_points[point_index * dimension + dim]);
                        min_target_box[dim] = std::min(min_target_box[dim], m_target_points[point_index * dimension + dim]);
                    }
                }
            }
        }
        for (int j = 0; j < N; j++) {
            const std::vector<int> &source_elt_indices = m_source_dofs_to_elements.at(cols[j]);
            int number_of_elements_sharing_dof         = source_elt_indices.size() / 2;
            for (int l = 0; l < number_of_elements_sharing_dof; l++) {
                source_element_indices.insert(source_elt_indices[l * 2]);
                source_element_indices_to_dofs[std::make_pair(source_elt_indices[l * 2], source_elt_indices[l * 2 + 1])] = j;
                for (int p = 0; p < m_source_number_of_points_per_element; p++) {
                    int point_index = m_source_elements_to_points[source_elt_indices[l * 2] * m_source_number_of_points_per_element + p];
                    for (int dim = 0; dim < dimension; dim++) {
                        max_source_box[dim] = std::max(max_source_box[dim], m_source_points[point_index * dimension + dim]);
                        min_source_box[dim] = std::min(min_source_box[dim], m_source_points[point_index * dimension + dim]);
                    }
                }
            }
        }

        // Check boxes
        for (int dim = 0; dim < dimension; dim++) {
            if (std ::abs(max_target_box[dim] - min_target_box[dim]) < std::numeric_limits<CoordinatePrecision>::epsilon() * 100) {
                max_target_box[dim] += 1; // std::numeric_limits<CoordinatePrecision>::epsilon() * 10000;
                min_target_box[dim] -= 1; // std::numeric_limits<CoordinatePrecision>::epsilon() * 10000;
            }

            if (std ::abs(max_source_box[dim] - min_source_box[dim]) < std::numeric_limits<CoordinatePrecision>::epsilon() * 100) {
                max_source_box[dim] += 1; // std::numeric_limits<CoordinatePrecision>::epsilon() * 10000;
                min_source_box[dim] -= 1; // std::numeric_limits<CoordinatePrecision>::epsilon() * 10000;
            }
        }

        // Compute target dof to interpolation matrix
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
            std::vector<CoefficientPrecision> weights;
            Matrix<CoefficientPrecision> tmp_mat(std::pow(L, dimension), m_target_number_of_points_per_element);
            if (m_target_number_of_points_per_element == 2) {
                const auto &rule = find_best_rule(m_target_quadrature_order, gauss_legendre_rules<CoordinatePrecision>);
                weights.resize(rule.nb_points);
                for (int i = 0; i < rule.nb_points; i++) {
                    weights[i] = rule.quad_points[i].w;
                }
                jacobian                       = segment_jacobian(tmp_cell_points[0], tmp_cell_points[1]);
                local_target_quadrature_points = map_reference_to_segment(tmp_cell_points[0], tmp_cell_points[1], rule);
            } else if (m_target_number_of_points_per_element == 3) {
                const auto &rule = find_best_rule(m_target_quadrature_order, triangle_rules<CoordinatePrecision>);
                weights.resize(rule.nb_points);
                for (int i = 0; i < rule.nb_points; i++) {
                    weights[i] = rule.quad_points[i].w;
                }
                jacobian                       = triangle_jacobian(tmp_cell_points[0], tmp_cell_points[1], tmp_cell_points[2]);
                local_target_quadrature_points = map_reference_to_triangle(tmp_cell_points[0], tmp_cell_points[1], tmp_cell_points[2], rule);
            }
            CoefficientPrecision *S_data;
            theia::get_polynomials<dimension, CoordinatePrecision, CoefficientPrecision, 0>(L, S_data, min_target_box.data(), max_target_box.data(), local_target_quadrature_points.data(), local_target_quadrature_points.size());
            Matrix<CoefficientPrecision> S;
            S.assign(std::pow(L, dimension), local_target_quadrature_points.size(), S_data, true);
            Matrix<CoefficientPrecision> phi(local_target_quadrature_points.size(), m_target_number_of_dofs_per_element);
            for (int i = 0; i < phi.nb_rows(); i++) {
                for (int j = 0; j < phi.nb_cols(); j++) {
                    phi(i, j) = jacobian * weights[i] * m_target_basis_function(tmp_cell_points, j, local_target_quadrature_points[i]);
                }
            }
            add_matrix_matrix_product('N', 'N', CoefficientPrecision(1), S, phi, CoefficientPrecision(0), tmp_mat);
            for (int i = 0; i < tmp_mat.nb_rows(); i++) {
                for (int j = 0; j < m_target_number_of_dofs_per_element; j++) {
                    U_tilde(target_element_indices_to_dofs[{target_element_index, j}], i) += tmp_mat(i, j);
                }
            }
        }

        // Compute source dof to interpolation matrix
        std::vector<std::array<double, dimension>> local_source_quadrature_points;
        tmp_cell_points.resize(m_source_number_of_points_per_element);
        for (auto source_element_index : source_element_indices) {
            for (int p = 0; p < m_source_number_of_points_per_element; p++) {
                int point_index = m_source_elements_to_points[source_element_index * m_source_number_of_points_per_element + p];
                for (int dim = 0; dim < dimension; dim++) {
                    tmp_cell_points[p][dim] = m_source_points[point_index * dimension + dim];
                }
            }
            CoordinatePrecision jacobian = 0;
            std::vector<CoefficientPrecision> weights;
            Matrix<CoefficientPrecision> tmp_mat(std::pow(L, dimension), m_source_number_of_points_per_element);
            if (m_source_number_of_points_per_element == 3) {
                const auto &rule = find_best_rule(m_source_quadrature_order, triangle_rules<CoordinatePrecision>);
                weights.resize(rule.nb_points);
                for (int i = 0; i < rule.nb_points; i++) {
                    weights[i] = rule.quad_points[i].w;
                }
                jacobian                       = triangle_jacobian(tmp_cell_points[0], tmp_cell_points[1], tmp_cell_points[2]);
                local_source_quadrature_points = map_reference_to_triangle(tmp_cell_points[0], tmp_cell_points[1], tmp_cell_points[2], rule);

            } else if (m_source_number_of_points_per_element == 2) {
                const auto &rule = find_best_rule(m_source_quadrature_order, gauss_legendre_rules<CoordinatePrecision>);
                weights.resize(rule.nb_points);
                for (int i = 0; i < rule.nb_points; i++) {
                    weights[i] = rule.quad_points[i].w;
                }
                jacobian                       = segment_jacobian(tmp_cell_points[0], tmp_cell_points[1]);
                local_source_quadrature_points = map_reference_to_segment(tmp_cell_points[0], tmp_cell_points[1], rule);
            }
            CoefficientPrecision *S_data;
            theia::get_polynomials<dimension, CoordinatePrecision, CoefficientPrecision, 0>(L, S_data, min_source_box.data(), max_source_box.data(), local_source_quadrature_points.data(), local_source_quadrature_points.size());
            Matrix<CoefficientPrecision> S;
            S.assign(std::pow(L, dimension), local_source_quadrature_points.size(), S_data, true);
            Matrix<CoefficientPrecision> phi(local_source_quadrature_points.size(), m_source_number_of_dofs_per_element);
            for (int i = 0; i < phi.nb_rows(); i++) {
                for (int j = 0; j < phi.nb_cols(); j++) {
                    phi(i, j) = jacobian * weights[i] * m_source_basis_function(tmp_cell_points, j, local_source_quadrature_points[i]);
                }
            }
            add_matrix_matrix_product('N', 'N', CoefficientPrecision(1), S, phi, CoefficientPrecision(0), tmp_mat);
            for (int i = 0; i < tmp_mat.nb_rows(); i++) {
                for (int j = 0; j < m_source_number_of_dofs_per_element; j++) {
                    V_tilde(i, source_element_indices_to_dofs[{source_element_index, j}]) += tmp_mat(i, j);
                }
            }
        }

        // Compute interpolation matrix
        std::vector<std::array<CoordinatePrecision, dimension>> px(std::pow(L, dimension)), py(std::pow(L, dimension));
        theia::get_multivariate_interp_nodes<dimension, CoordinatePrecision, 0>(L, min_target_box.data(), max_target_box.data(), px.data());
        theia::get_multivariate_interp_nodes<dimension, CoordinatePrecision, 0>(L, min_source_box.data(), max_source_box.data(), py.data());
        theia::get_symbolic_matrix<dimension, CoordinatePrecision, CoefficientPrecision, kernel_type>(px.data(), py.data(), pow(L, dimension), pow(L, dimension), K.data(), &m_kernel);

        // truncated SVD
        std::vector<underlying_type<CoefficientPrecision>> singular_values(std::pow(L, dimension));
        Matrix<CoefficientPrecision> u(std::pow(L, dimension), std::pow(L, dimension));
        Matrix<CoefficientPrecision> vt(std::pow(L, dimension), std::pow(L, dimension));
        int truncated_rank = SVD_truncation(K, epsilon, u, vt, singular_values);
        if (reqrank > 0)
            truncated_rank = std::min(reqrank, std::min(M, N));
        Matrix<CoefficientPrecision> truncated_u(std::pow(L, dimension), truncated_rank);
        Matrix<CoefficientPrecision> truncated_vt(truncated_rank, std::pow(L, dimension));
        for (int i = 0; i < std::pow(L, dimension); i++) {
            for (int j = 0; j < truncated_rank; j++) {
                truncated_u(i, j) = u(i, j) * singular_values[j];
            }
        }
        for (int i = 0; i < truncated_rank; i++) {
            for (int j = 0; j < std::pow(L, dimension); j++) {
                truncated_vt(i, j) = vt(i, j);
            }
        }

        // Set lrmat
        auto &U = lrmat.get_U();
        auto &V = lrmat.get_V();
        U.resize(M, truncated_rank);
        V.resize(truncated_rank, N);
        add_matrix_matrix_product('N', 'N', CoefficientPrecision(1), U_tilde, truncated_u, CoefficientPrecision(0), U);
        add_matrix_matrix_product('N', 'N', CoefficientPrecision(1), truncated_vt, V_tilde, CoefficientPrecision(0), V);

        return true;
    }

  public:
    BEMHCA(kernel_type kernel, basis_function_type target_basis_function, std::map<int, std::vector<int>> target_dofs_to_elements, int target_number_of_dofs_per_element, const int *target_elements_to_points, int target_number_of_points_per_element, CoordinatePrecision *target_points, int target_size, const int *target_permutation, int target_quadrature_order, basis_function_type source_basis_function, std::map<int, std::vector<int>> source_dofs_to_elements, int source_number_of_dofs_per_element, int *source_elements_to_points, int source_number_of_points_per_element, CoordinatePrecision *source_points, int source_size, const int *source_permutation, int source_quadrature_order) : m_kernel(kernel), m_target_basis_function(target_basis_function), m_target_dofs_to_elements(target_dofs_to_elements), m_target_number_of_dofs_per_element(target_number_of_dofs_per_element), m_target_elements_to_points(target_elements_to_points), m_target_number_of_points_per_element(target_number_of_points_per_element), m_target_points(target_points), m_target_size(target_size), m_target_permutation(target_permutation), m_target_quadrature_order(target_quadrature_order), m_source_basis_function(source_basis_function), m_source_dofs_to_elements(source_dofs_to_elements), m_source_number_of_dofs_per_element(source_number_of_dofs_per_element), m_source_elements_to_points(source_elements_to_points), m_source_number_of_points_per_element(source_number_of_points_per_element), m_source_points(source_points), m_source_size(source_size), m_source_permutation(source_permutation), m_source_quadrature_order(source_quadrature_order) {}

    bool copy_low_rank_approximation(int M, int N, int row_offset, int col_offset, LowRankMatrix<CoefficientPrecision> &lrmat) const override {
        int reqrank = -1;
        return copy_low_rank_approximation(M, N, m_target_permutation + row_offset, m_source_permutation + col_offset, lrmat.get_epsilon(), reqrank, lrmat);
    }

    bool copy_low_rank_approximation(int M, int N, int row_offset, int col_offset, int reqrank, LowRankMatrix<CoefficientPrecision> &lrmat) const override {
        return copy_low_rank_approximation(M, N, m_target_permutation + row_offset, m_source_permutation + col_offset, lrmat.get_epsilon(), reqrank, lrmat);
    }
};

} // namespace htool

#endif
