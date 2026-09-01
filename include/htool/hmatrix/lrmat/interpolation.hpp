#ifndef HTOOL_VIRTUAL_IBCOMP_HPP
#define HTOOL_VIRTUAL_IBCOMP_HPP

#include "../../hmatrix/interfaces/virtual_lrmat_generator.hpp" // for Virt...
#include "../../quadratures/geometry.hpp"
#include "htool/matrix/linalg/add_matrix_matrix_product.hpp"
#include "htool/matrix/utils/SVD_truncation.hpp"
#include "htool/misc/fem_interface.hpp"
#include "htool/misc/misc.hpp"
#include "htool/quadratures/gauss_legendre.hpp"
#include "htool/quadratures/triangle.hpp"
#include <functional>
#include <map>
#include <set>
#include <stdexcept>
#include <utility>
#include <variant>

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
            max_target_box[dim] = std::numeric_limits<CoordinatePrecision>::lowest();
            min_target_box[dim] = std::numeric_limits<CoordinatePrecision>::max();
            max_source_box[dim] = std::numeric_limits<CoordinatePrecision>::lowest();
            min_source_box[dim] = std::numeric_limits<CoordinatePrecision>::max();
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
            max_target_box[dim] = std::numeric_limits<CoordinatePrecision>::lowest();
            min_target_box[dim] = std::numeric_limits<CoordinatePrecision>::max();
            max_source_box[dim] = std::numeric_limits<CoordinatePrecision>::lowest();
            min_source_box[dim] = std::numeric_limits<CoordinatePrecision>::max();
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

    bool check_size = false;

  private:
    kernel_type m_kernel;

    // Quadrature rule, reference points and weights for one side (target or source). Determined
    // entirely by quadrature_order/number_of_points_per_element, which are fixed for the lifetime of
    // a BEMHCA - so this is built once, in the constructor, and never mutated afterward. Concurrent
    // const reads from multiple threads (copy_low_rank_approximation runs from H-matrix assembly's
    // threaded loop) are safe precisely because nothing writes to it after construction.
    struct interpolation_setup {
        std::variant<GaussLegendreRule<CoordinatePrecision>, TriangleRule<CoordinatePrecision>> rule;
        std::vector<std::vector<CoordinatePrecision>> reference_quadrature_points;
        std::vector<CoefficientPrecision> weights;

        interpolation_setup(int quadrature_order, int number_of_points_per_element) {
            if (number_of_points_per_element == 2) {
                rule = find_best_rule(quadrature_order, gauss_legendre_rules<CoordinatePrecision>);
            } else if (number_of_points_per_element == 3) {
                rule = find_best_rule(quadrature_order, triangle_rules<CoordinatePrecision>);
            } else {
                throw std::invalid_argument("interpolation_setup: unsupported number_of_points_per_element = " + std::to_string(number_of_points_per_element));
            }
            std::visit([this](const auto &r) {
                reference_quadrature_points.assign(r.nb_points, std::vector<CoordinatePrecision>(r.quad_points[0].point.size()));
                weights.resize(r.nb_points);
                for (std::size_t i = 0; i < r.nb_points; i++) {
                    for (std::size_t d = 0; d < r.quad_points[i].point.size(); d++)
                        reference_quadrature_points[i][d] = r.quad_points[i].point[d];
                    weights[i] = r.quad_points[i].w;
                }
            },
                       rule);
        }
    };

    // Per-element working buffers, built fresh for each copy_low_rank_approximation call (cheap:
    // just two resize()s, sized from the shared interpolation_setup) and never shared across calls -
    // safe to use concurrently from different threads on different blocks.
    struct interpolation_scratch {
        std::vector<std::array<CoordinatePrecision, dimension>> local_quadrature_points;
        std::vector<std::array<CoordinatePrecision, dimension>> local_cell_points;

        interpolation_scratch(int number_of_points_per_element, std::size_t nb_quadrature_points) {
            local_cell_points.resize(number_of_points_per_element);
            local_quadrature_points.resize(nb_quadrature_points);
        }

        CoordinatePrecision map_cell_to_quadrature(const interpolation_setup &setup) {
            if (auto *segment_rule = std::get_if<GaussLegendreRule<CoordinatePrecision>>(&setup.rule)) {
                map_reference_to_segment(local_cell_points[0], local_cell_points[1], *segment_rule, local_quadrature_points);
                return segment_jacobian(local_cell_points[0], local_cell_points[1]);
            }
            const auto &triangle_rule = std::get<TriangleRule<CoordinatePrecision>>(setup.rule);
            map_reference_to_triangle(local_cell_points[0], local_cell_points[1], local_cell_points[2], triangle_rule, local_quadrature_points);
            return triangle_jacobian(local_cell_points[0], local_cell_points[1], local_cell_points[2]);
        }
    };

    FEMSpace<CoefficientPrecision, CoordinatePrecision, dimension> m_target;
    FEMSpace<CoefficientPrecision, CoordinatePrecision, dimension> m_source;
    interpolation_setup m_target_interpolation_setup;
    interpolation_setup m_source_interpolation_setup;

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
        int n_interp = 1;
        for (int d = 0; d < dimension; d++)
            n_interp *= L;
        Matrix<CoefficientPrecision> U_tilde(M, n_interp);
        Matrix<CoefficientPrecision> V_tilde(n_interp, N);
        Matrix<CoefficientPrecision> K(n_interp, n_interp);

        // Compute boxes
        for (int dim = 0; dim < dimension; dim++) {
            max_target_box[dim] = std::numeric_limits<CoordinatePrecision>::lowest();
            min_target_box[dim] = std::numeric_limits<CoordinatePrecision>::max();
            max_source_box[dim] = std::numeric_limits<CoordinatePrecision>::lowest();
            min_source_box[dim] = std::numeric_limits<CoordinatePrecision>::max();
        }
        for (int i = 0; i < M; i++) {
            const std::vector<int> &target_elt_indices = m_target.dofs_to_elements->at(rows[i]);
            int number_of_elements_sharing_dof         = target_elt_indices.size() / 2;
            for (int l = 0; l < number_of_elements_sharing_dof; l++) {
                target_element_indices.insert(target_elt_indices[l * 2]);
                target_element_indices_to_dofs[std::make_pair(target_elt_indices[l * 2], target_elt_indices[l * 2 + 1])] = i;

                for (int p = 0; p < m_target.number_of_points_per_element; p++) {

                    int point_index = m_target.elements_to_points[target_elt_indices[l * 2] * m_target.number_of_points_per_element + p];
                    for (int dim = 0; dim < dimension; dim++) {
                        max_target_box[dim] = std::max(max_target_box[dim], m_target.points[point_index * dimension + dim]);
                        min_target_box[dim] = std::min(min_target_box[dim], m_target.points[point_index * dimension + dim]);
                    }
                }
            }
        }
        for (int j = 0; j < N; j++) {
            const std::vector<int> &source_elt_indices = m_source.dofs_to_elements->at(cols[j]);
            int number_of_elements_sharing_dof         = source_elt_indices.size() / 2;
            for (int l = 0; l < number_of_elements_sharing_dof; l++) {
                source_element_indices.insert(source_elt_indices[l * 2]);
                source_element_indices_to_dofs[std::make_pair(source_elt_indices[l * 2], source_elt_indices[l * 2 + 1])] = j;
                for (int p = 0; p < m_source.number_of_points_per_element; p++) {
                    int point_index = m_source.elements_to_points[source_elt_indices[l * 2] * m_source.number_of_points_per_element + p];
                    for (int dim = 0; dim < dimension; dim++) {
                        max_source_box[dim] = std::max(max_source_box[dim], m_source.points[point_index * dimension + dim]);
                        min_source_box[dim] = std::min(min_source_box[dim], m_source.points[point_index * dimension + dim]);
                    }
                }
            }
        }

        // Check box intersection
        bool are_box_intersecting = true;
        for (int dim = 0; dim < dimension; dim++) {
            if (are_box_intersecting && (max_target_box[dim] < min_source_box[dim] || max_source_box[dim] < min_target_box[dim])) {
                are_box_intersecting = false;
            }
        }
        if (are_box_intersecting) {
            return false;
        }

        // Check boxes
        for (int dim = 0; dim < dimension; dim++) {
            //
            max_source_box[dim] += 0.001;
            min_source_box[dim] -= 0.001;
            max_target_box[dim] += 0.001;
            min_target_box[dim] -= 0.001;
            if (std ::abs(max_target_box[dim] - min_target_box[dim]) < std::numeric_limits<CoordinatePrecision>::epsilon() * 100) {
                max_target_box[dim] += 0.01; // std::numeric_limits<CoordinatePrecision>::epsilon() * 10000;
                min_target_box[dim] -= 0.01; // std::numeric_limits<CoordinatePrecision>::epsilon() * 10000;
            }

            if (std ::abs(max_source_box[dim] - min_source_box[dim]) < std::numeric_limits<CoordinatePrecision>::epsilon() * 100) {
                max_source_box[dim] += 0.01; // std::numeric_limits<CoordinatePrecision>::epsilon() * 10000;
                min_source_box[dim] -= 0.01; // std::numeric_limits<CoordinatePrecision>::epsilon() * 10000;
            }
        }

        // Check box intersection
        are_box_intersecting = true;
        for (int dim = 0; dim < dimension; dim++) {
            if (are_box_intersecting && (max_target_box[dim] < min_source_box[dim] || max_source_box[dim] < min_target_box[dim])) {
                are_box_intersecting = false;
            }
        }
        if (are_box_intersecting) {
            return false;
        }

        // Compute target dof to interpolation matrix
        interpolation_scratch target_scratch(m_target.number_of_points_per_element, m_target_interpolation_setup.reference_quadrature_points.size());
        Matrix<CoefficientPrecision> tmp_mat(n_interp, m_target.number_of_dofs_per_element);
        Matrix<CoefficientPrecision> S(n_interp, target_scratch.local_quadrature_points.size());
        Matrix<CoefficientPrecision> phi(target_scratch.local_quadrature_points.size(), m_target.number_of_dofs_per_element);
        for (auto target_element_index : target_element_indices) {
            compute_elementwise_interpolation_matrix(L, target_element_index, m_target, m_target_interpolation_setup, target_scratch, min_target_box, max_target_box, S, phi, tmp_mat);
            for (int i = 0; i < tmp_mat.nb_rows(); i++) {
                for (int j = 0; j < m_target.number_of_dofs_per_element; j++) {
                    U_tilde(target_element_indices_to_dofs[{target_element_index, j}], i) += tmp_mat(i, j);
                }
            }
        }

        // Compute source dof to interpolation matrix
        interpolation_scratch source_scratch(m_source.number_of_points_per_element, m_source_interpolation_setup.reference_quadrature_points.size());
        for (auto source_element_index : source_element_indices) {
            compute_elementwise_interpolation_matrix(L, source_element_index, m_source, m_source_interpolation_setup, source_scratch, min_source_box, max_source_box, S, phi, tmp_mat);
            for (int i = 0; i < tmp_mat.nb_rows(); i++) {
                for (int j = 0; j < m_source.number_of_dofs_per_element; j++) {
                    V_tilde(i, source_element_indices_to_dofs[{source_element_index, j}]) += tmp_mat(i, j);
                }
            }
        }

        // Compute interpolation matrix
        std::vector<std::array<CoordinatePrecision, dimension>> px(n_interp), py(n_interp);
        theia::get_multivariate_interp_nodes<dimension, CoordinatePrecision, 0>(L, min_target_box.data(), max_target_box.data(), px.data());
        theia::get_multivariate_interp_nodes<dimension, CoordinatePrecision, 0>(L, min_source_box.data(), max_source_box.data(), py.data());
        theia::get_symbolic_matrix<dimension, CoordinatePrecision, CoefficientPrecision, kernel_type>(px.data(), py.data(), n_interp, n_interp, K.data(), &m_kernel);

        // truncated SVD
        std::vector<underlying_type<CoefficientPrecision>> singular_values(n_interp);
        Matrix<CoefficientPrecision> u(n_interp, n_interp);
        Matrix<CoefficientPrecision> vt(n_interp, n_interp);
        int truncated_rank = SVD_truncation(K, epsilon, u, vt, singular_values);
        if (reqrank > 0)
            truncated_rank = std::min({reqrank, n_interp, M, N});
        if (check_size && (truncated_rank * (M + N) > (M * N))) {
            return false;
        }
        Matrix<CoefficientPrecision> truncated_u(n_interp, truncated_rank);
        Matrix<CoefficientPrecision> truncated_vt(truncated_rank, n_interp);
        for (int i = 0; i < n_interp; i++) {
            for (int j = 0; j < truncated_rank; j++) {
                truncated_u(i, j) = u(i, j) * singular_values[j];
            }
        }
        for (int i = 0; i < truncated_rank; i++) {
            for (int j = 0; j < n_interp; j++) {
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

    // min_box/max_box are taken by non-const reference (even though they aren't modified here)
    // because theia::get_polynomials expects non-const FLT* mins/maxs; it only reads through them,
    // but its signature isn't const-correct, and the callers' boxes are plain (non-const) locals.
    void compute_elementwise_interpolation_matrix(int L, int element_index, const FEMSpace<CoefficientPrecision, CoordinatePrecision, dimension> &fe_space, const interpolation_setup &setup, interpolation_scratch &scratch, std::array<CoordinatePrecision, dimension> &min_box, std::array<CoordinatePrecision, dimension> &max_box, Matrix<CoefficientPrecision> &S, Matrix<CoefficientPrecision> &phi, Matrix<CoefficientPrecision> &tmp_mat) const {
        int n_interp = 1;
        for (int d = 0; d < dimension; d++)
            n_interp *= L;

        // Check size
        if (tmp_mat.nb_rows() != n_interp || tmp_mat.nb_cols() != fe_space.number_of_dofs_per_element)
            tmp_mat.resize(n_interp, fe_space.number_of_dofs_per_element);
        if (S.nb_rows() != n_interp || S.nb_cols() != scratch.local_quadrature_points.size())
            S.resize(n_interp, scratch.local_quadrature_points.size());
        if (phi.nb_rows() != scratch.local_quadrature_points.size() || phi.nb_cols() != fe_space.number_of_dofs_per_element)
            phi.resize(scratch.local_quadrature_points.size(), fe_space.number_of_dofs_per_element);

        // Compute S*phi for a given element
        for (int p = 0; p < fe_space.number_of_points_per_element; p++) {
            int point_index = fe_space.elements_to_points[element_index * fe_space.number_of_points_per_element + p];
            for (int dim = 0; dim < dimension; dim++) {
                scratch.local_cell_points[p][dim] = fe_space.points[point_index * dimension + dim];
            }
        }
        CoordinatePrecision jacobian = scratch.map_cell_to_quadrature(setup);
        CoefficientPrecision *S_ptr  = S.data();
        theia::get_polynomials<dimension, CoordinatePrecision, CoefficientPrecision, 0>(L, S_ptr, min_box.data(), max_box.data(), scratch.local_quadrature_points.data(), scratch.local_quadrature_points.size());
        for (int i = 0; i < phi.nb_rows(); i++) {
            for (int j = 0; j < phi.nb_cols(); j++) {
                phi(i, j) = jacobian * setup.weights[i] * fe_space.basis_function(element_index, j, setup.reference_quadrature_points[i]);
            }
        }
        add_matrix_matrix_product('N', 'N', CoefficientPrecision(1), S, phi, CoefficientPrecision(0), tmp_mat);
    }

  public:
    BEMHCA(kernel_type kernel, FEMSpace<CoefficientPrecision, CoordinatePrecision, dimension> target, FEMSpace<CoefficientPrecision, CoordinatePrecision, dimension> source) : m_kernel(kernel), m_target(std::move(target)), m_source(std::move(source)), m_target_interpolation_setup(m_target.quadrature_order, m_target.number_of_points_per_element), m_source_interpolation_setup(m_source.quadrature_order, m_source.number_of_points_per_element) {}

    bool copy_low_rank_approximation(int M, int N, int row_offset, int col_offset, LowRankMatrix<CoefficientPrecision> &lrmat) const override {
        int reqrank = -1;
        return copy_low_rank_approximation(M, N, m_target.permutation + row_offset, m_source.permutation + col_offset, lrmat.get_epsilon(), reqrank, lrmat);
    }

    bool copy_low_rank_approximation(int M, int N, int row_offset, int col_offset, int reqrank, LowRankMatrix<CoefficientPrecision> &lrmat) const override {
        return copy_low_rank_approximation(M, N, m_target.permutation + row_offset, m_source.permutation + col_offset, lrmat.get_epsilon(), reqrank, lrmat);
    }
};

} // namespace htool

#endif
