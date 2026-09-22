#ifndef HTOOL_VIRTUAL_IBCOMP_HPP
#define HTOOL_VIRTUAL_IBCOMP_HPP

#include "../../hmatrix/interfaces/virtual_lrmat_generator.hpp" // for Virt...
#include "../../quadratures/geometry.hpp"
#include "htool/basic_types/tree.hpp"
#include "htool/clustering/cluster_node.hpp"
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

    bool check_size       = false;
    bool precomputed_PtoM = false;

  private:
    kernel_type m_kernel;
    std::map<const Cluster<CoordinatePrecision> *, Matrix<CoefficientPrecision>> m_target_interpolated_leaves;
    std::map<const Cluster<CoordinatePrecision> *, Matrix<CoefficientPrecision>> m_source_interpolated_leaves;

    // Quadrature rule, reference points and weights for one side (target or source). Built once in
    // the constructor and never mutated after, so concurrent const reads from multiple threads
    // (copy_low_rank_approximation runs from H-matrix assembly's threaded loop) are safe.
    //
    // is_segment_rule/segment_rule/triangle_rule stand in for a std::variant (htool stays C++14): the
    // active rule is fixed at construction, so a flag suffices, no tagged union needed.
    struct interpolation_setup {
        bool is_segment_rule;
        GaussLegendreRule<CoordinatePrecision> segment_rule;
        TriangleRule<CoordinatePrecision> triangle_rule;
        std::vector<std::vector<CoordinatePrecision>> reference_quadrature_points;
        std::vector<CoefficientPrecision> weights;

        interpolation_setup(int quadrature_order, int number_of_points_per_element) : is_segment_rule(number_of_points_per_element == 2) {
            if (number_of_points_per_element == 2) {
                segment_rule = find_best_rule(quadrature_order, gauss_legendre_rules<CoordinatePrecision>);
                fill_reference_quadrature(segment_rule);
            } else if (number_of_points_per_element == 3) {
                triangle_rule = find_best_rule(quadrature_order, triangle_rules<CoordinatePrecision>);
                fill_reference_quadrature(triangle_rule);
            } else {
                throw std::invalid_argument("interpolation_setup: unsupported number_of_points_per_element = " + std::to_string(number_of_points_per_element));
            }
        }

      private:
        template <typename Rule>
        void fill_reference_quadrature(const Rule &r) {
            reference_quadrature_points.assign(r.nb_points, std::vector<CoordinatePrecision>(r.quad_points[0].point.size()));
            weights.resize(r.nb_points);
            for (std::size_t i = 0; i < r.nb_points; i++) {
                for (std::size_t d = 0; d < r.quad_points[i].point.size(); d++)
                    reference_quadrature_points[i][d] = r.quad_points[i].point[d];
                weights[i] = r.quad_points[i].w;
            }
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
            if (setup.is_segment_rule) {
                map_reference_to_segment(local_cell_points[0], local_cell_points[1], setup.segment_rule, local_quadrature_points);
                return segment_jacobian(local_cell_points[0], local_cell_points[1]);
            }
            map_reference_to_triangle(local_cell_points[0], local_cell_points[1], local_cell_points[2], setup.triangle_rule, local_quadrature_points);
            return triangle_jacobian(local_cell_points[0], local_cell_points[1], local_cell_points[2]);
        }
    };

    FEMSpace<CoefficientPrecision, CoordinatePrecision, dimension> m_target;
    FEMSpace<CoefficientPrecision, CoordinatePrecision, dimension> m_source;
    interpolation_setup m_target_interpolation_setup;
    interpolation_setup m_source_interpolation_setup;

    // Bounding box of the elements incident to `size` dofs, plus the (element, local_dof) -> local
    // row/col map needed to accumulate an interpolation matrix.
    struct side_geometry {
        std::array<CoordinatePrecision, dimension> min_box, max_box;
        std::set<int> element_indices;
        std::map<std::pair<int, int>, int> element_indices_to_dofs;
    };

    side_geometry collect_side(int size, const int *dofs, const FEMSpace<CoefficientPrecision, CoordinatePrecision, dimension> &fe_space) const {
        side_geometry geom;
        for (int dim = 0; dim < dimension; dim++) {
            geom.max_box[dim] = std::numeric_limits<CoordinatePrecision>::lowest();
            geom.min_box[dim] = std::numeric_limits<CoordinatePrecision>::max();
        }
        for (int i = 0; i < size; i++) {
            const std::vector<int> &elt_indices = fe_space.dofs_to_elements->at(dofs[i]);
            int number_of_elements_sharing_dof  = elt_indices.size() / 2;
            for (int l = 0; l < number_of_elements_sharing_dof; l++) {
                geom.element_indices.insert(elt_indices[l * 2]);
                geom.element_indices_to_dofs[std::make_pair(elt_indices[l * 2], elt_indices[l * 2 + 1])] = i;
                for (int p = 0; p < fe_space.number_of_points_per_element; p++) {
                    int point_index = fe_space.elements_to_points[elt_indices[l * 2] * fe_space.number_of_points_per_element + p];
                    for (int dim = 0; dim < dimension; dim++) {
                        geom.max_box[dim] = std::max(geom.max_box[dim], fe_space.points[point_index * dimension + dim]);
                        geom.min_box[dim] = std::min(geom.min_box[dim], fe_space.points[point_index * dimension + dim]);
                    }
                }
            }
        }
        return geom;
    }

    // Pad a box by a fixed margin, and inflate further if it is degenerate along some axis.
    static void inflate_box(std::array<CoordinatePrecision, dimension> &min_box, std::array<CoordinatePrecision, dimension> &max_box) {
        for (int dim = 0; dim < dimension; dim++) {
            max_box[dim] += 0.001;
            min_box[dim] -= 0.001;
            if (std::abs(max_box[dim] - min_box[dim]) < std::numeric_limits<CoordinatePrecision>::epsilon() * 100) {
                max_box[dim] += 0.01;
                min_box[dim] -= 0.01;
            }
        }
    }

    // Dof-to-interpolation-node matrix (n_interp x size) for one side, accumulated element by element.
    Matrix<CoefficientPrecision> build_interpolation_matrix(int L, int n_interp, int size, const side_geometry &geom, const FEMSpace<CoefficientPrecision, CoordinatePrecision, dimension> &fe_space, const interpolation_setup &setup) const {
        Matrix<CoefficientPrecision> dof_to_interp(n_interp, size);
        interpolation_scratch scratch(fe_space.number_of_points_per_element, setup.reference_quadrature_points.size());
        Matrix<CoefficientPrecision> tmp_mat(n_interp, fe_space.number_of_dofs_per_element);
        Matrix<CoefficientPrecision> S(n_interp, scratch.local_quadrature_points.size());
        Matrix<CoefficientPrecision> phi(scratch.local_quadrature_points.size(), fe_space.number_of_dofs_per_element);
        std::array<CoordinatePrecision, dimension> min_box = geom.min_box, max_box = geom.max_box; // compute_elementwise_interpolation_matrix wants non-const refs

        for (auto element_index : geom.element_indices) {
            compute_elementwise_interpolation_matrix(L, element_index, fe_space, setup, scratch, min_box, max_box, S, phi, tmp_mat);
            for (int i = 0; i < tmp_mat.nb_rows(); i++) {
                for (int j = 0; j < fe_space.number_of_dofs_per_element; j++) {
                    auto it = geom.element_indices_to_dofs.find(std::make_pair(element_index, j));
                    if (it != geom.element_indices_to_dofs.end()) {
                        dof_to_interp(i, it->second) += tmp_mat(i, j);
                    }
                }
            }
        }
        return dof_to_interp;
    }

    bool copy_low_rank_approximation(int M, int N, const int *rows, const int *cols, underlying_type<CoefficientPrecision> epsilon, int reqrank, LowRankMatrix<CoefficientPrecision> &lrmat) const {
        int L        = std::ceil(std::log(1. / epsilon) / std::log(10)) + 1;
        int n_interp = 1;
        for (int d = 0; d < dimension; d++)
            n_interp *= L;

        side_geometry target_geom = collect_side(M, rows, m_target);
        side_geometry source_geom = collect_side(N, cols, m_source);

        // Check box intersection
        bool are_box_intersecting = true;
        for (int dim = 0; dim < dimension; dim++) {
            if (are_box_intersecting && (target_geom.max_box[dim] < source_geom.min_box[dim] || source_geom.max_box[dim] < target_geom.min_box[dim])) {
                are_box_intersecting = false;
            }
        }
        if (are_box_intersecting) {
            return false;
        }

        inflate_box(target_geom.min_box, target_geom.max_box);
        inflate_box(source_geom.min_box, source_geom.max_box);

        // Check box intersection
        are_box_intersecting = true;
        for (int dim = 0; dim < dimension; dim++) {
            if (are_box_intersecting && (target_geom.max_box[dim] < source_geom.min_box[dim] || source_geom.max_box[dim] < target_geom.min_box[dim])) {
                are_box_intersecting = false;
            }
        }
        if (are_box_intersecting) {
            return false;
        }

        // Compute target/source dof to interpolation matrix, n_interp x size each
        Matrix<CoefficientPrecision> U_tilde = build_interpolation_matrix(L, n_interp, M, target_geom, m_target, m_target_interpolation_setup);
        Matrix<CoefficientPrecision> V_tilde = build_interpolation_matrix(L, n_interp, N, source_geom, m_source, m_source_interpolation_setup);

        // Compute interpolation matrix
        std::vector<std::array<CoordinatePrecision, dimension>> px(n_interp), py(n_interp);
        Matrix<CoefficientPrecision> K(n_interp, n_interp);
        theia::get_multivariate_interp_nodes<dimension, CoordinatePrecision, 0>(L, target_geom.min_box.data(), target_geom.max_box.data(), px.data());
        theia::get_multivariate_interp_nodes<dimension, CoordinatePrecision, 0>(L, source_geom.min_box.data(), source_geom.max_box.data(), py.data());
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
        add_matrix_matrix_product('T', 'N', CoefficientPrecision(1), U_tilde, truncated_u, CoefficientPrecision(0), U);
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
    BEMHCA(kernel_type kernel, FEMSpace<CoefficientPrecision, CoordinatePrecision, dimension> target, FEMSpace<CoefficientPrecision, CoordinatePrecision, dimension> source) : m_kernel(kernel), m_target(std::move(target)), m_source(std::move(source)), m_target_interpolation_setup(m_target.quadrature_order, m_target.number_of_points_per_element), m_source_interpolation_setup(m_source.quadrature_order, m_source.number_of_points_per_element) {
    }

    // BEMHCA(kernel_type kernel, FEMSpace<CoefficientPrecision, CoordinatePrecision, dimension> target, const Cluster<CoordinatePrecision> &target_cluster, FEMSpace<CoefficientPrecision, CoordinatePrecision, dimension> source, const Cluster<CoordinatePrecision> &source_cluster, bool use_precomputed_PtoL) : m_kernel(kernel), m_target(std::move(target)), m_source(std::move(source)), m_target_interpolation_setup(m_target.quadrature_order, m_target.number_of_points_per_element), m_source_interpolation_setup(m_source.quadrature_order, m_source.number_of_points_per_element) {
    //     precomputed_PtoM = true;

    //     // Get cluster leaves
    //     preorder_tree_traversal(target_cluster,
    //                             [this](const Cluster<CoordinatePrecision> &current_cluster) {
    //                                 if (current_cluster.is_leaf()) {
    //                                     m_target_interpolated_leaves[&current_cluster];
    //                                 }
    //                             });
    //     preorder_tree_traversal(source_cluster,
    //                             [this](const Cluster<CoordinatePrecision> &current_cluster) {
    //                                 if (current_cluster.is_leaf()) {
    //                                     m_source_interpolated_leaves[&current_cluster];
    //                                 }
    //                             });

    //     // Interpolate leaves
    //     for (auto &target_leaf_pair : m_target_interpolated_leaves) {
    //         auto target_leaf_cluster           = target_leaf_pair.first;
    //         auto &target_leaf_interpolated_mat = target_leaf_pair.second;
    //     }
    // }

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
