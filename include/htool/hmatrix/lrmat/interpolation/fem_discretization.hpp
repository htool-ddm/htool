#ifndef HTOOL_HMATRIX_LRMAT_INTERPOLATION_FEM_DISCRETIZATION_HPP
#define HTOOL_HMATRIX_LRMAT_INTERPOLATION_FEM_DISCRETIZATION_HPP

#include "../../../matrix/linalg/add_matrix_matrix_product.hpp"
#include "../../../misc/fem_interface.hpp"
#include "../../../quadratures/gauss_legendre.hpp"
#include "../../../quadratures/geometry.hpp"
#include "../../../quadratures/triangle.hpp"
#include "discretization.hpp"
#include <map>
#include <set>
#include <stdexcept>
#include <string>

namespace htool {

// Finite element space: a dof's interpolation is the integral of the Lagrange polynomials against its basis
// function, computed by quadrature on the elements it touches. Copies the FEMSpace struct, not what it points
// to (dofs_to_elements, connectivity, points) nor the state basis_function captures: those, and the cluster,
// must outlive this object.
template <typename CoefficientPrecision, typename CoordinatePrecision, int dimension>
class FEMDiscretization final : public VirtualDiscretization<CoefficientPrecision, CoordinatePrecision, dimension> {
    // Quadrature rule, reference points and weights. Built once in the constructor and never mutated after, so
    // concurrent const reads are safe.
    //
    // is_segment_rule/segment_rule/triangle_rule stand in for a std::variant (htool stays C++14): the active rule
    // is fixed at construction, so a flag suffices, no tagged union needed.
    struct quadrature_setup {
        bool is_segment_rule;
        GaussLegendreRule<CoordinatePrecision> segment_rule;
        TriangleRule<CoordinatePrecision> triangle_rule;
        std::vector<std::vector<CoordinatePrecision>> reference_quadrature_points;
        std::vector<CoefficientPrecision> weights;

        quadrature_setup(int quadrature_order, int number_of_points_per_element) : is_segment_rule(number_of_points_per_element == 2) {
            if (number_of_points_per_element == 2) {
                segment_rule = find_best_rule(quadrature_order, gauss_legendre_rules<CoordinatePrecision>);
                fill_reference_quadrature(segment_rule);
            } else if (number_of_points_per_element == 3) {
                triangle_rule = find_best_rule(quadrature_order, triangle_rules<CoordinatePrecision>);
                fill_reference_quadrature(triangle_rule);
            } else {
                throw std::invalid_argument("quadrature_setup: unsupported number_of_points_per_element = " + std::to_string(number_of_points_per_element));
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

    // Per-element working buffers, built fresh for each call and never shared across calls.
    struct quadrature_scratch {
        std::vector<std::array<CoordinatePrecision, dimension>> local_quadrature_points;
        std::vector<std::array<CoordinatePrecision, dimension>> local_cell_points;

        quadrature_scratch(int number_of_points_per_element, std::size_t nb_quadrature_points) {
            local_cell_points.resize(number_of_points_per_element);
            local_quadrature_points.resize(nb_quadrature_points);
        }

        CoordinatePrecision map_cell_to_quadrature(const quadrature_setup &setup) {
            if (setup.is_segment_rule) {
                map_reference_to_segment(local_cell_points[0], local_cell_points[1], setup.segment_rule, local_quadrature_points);
                return segment_jacobian(local_cell_points[0], local_cell_points[1]);
            }
            map_reference_to_triangle(local_cell_points[0], local_cell_points[1], local_cell_points[2], setup.triangle_rule, local_quadrature_points);
            return triangle_jacobian(local_cell_points[0], local_cell_points[1], local_cell_points[2]);
        }
    };

    // Bounding box of the elements incident to a dof range, plus the (element, local_dof) -> local dof map
    // needed to accumulate an interpolation matrix.
    struct side_geometry {
        interpolation_box<CoordinatePrecision, dimension> box;
        std::set<int> element_indices;
        std::map<std::pair<int, int>, int> element_indices_to_dofs;
    };

    FEMSpace<CoefficientPrecision, CoordinatePrecision, dimension> m_space;
    const Cluster<CoordinatePrecision> *m_cluster;
    quadrature_setup m_setup;

    side_geometry collect_side(int offset, int size) const {
        side_geometry geom;
        geom.box.first.fill(std::numeric_limits<CoordinatePrecision>::max());
        geom.box.second.fill(std::numeric_limits<CoordinatePrecision>::lowest());
        for (int i = 0; i < size; i++) {
            const std::vector<int> &elt_indices = m_space.dofs_to_elements->at(m_cluster->get_permutation()[offset + i]);
            int number_of_elements_sharing_dof  = elt_indices.size() / 2;
            for (int l = 0; l < number_of_elements_sharing_dof; l++) {
                geom.element_indices.insert(elt_indices[l * 2]);
                geom.element_indices_to_dofs[std::make_pair(elt_indices[l * 2], elt_indices[l * 2 + 1])] = i;
                for (int p = 0; p < m_space.number_of_points_per_element; p++) {
                    int point_index = m_space.elements_to_points[elt_indices[l * 2] * m_space.number_of_points_per_element + p];
                    for (int dim = 0; dim < dimension; dim++) {
                        geom.box.second[dim] = std::max(geom.box.second[dim], m_space.points[point_index * dimension + dim]);
                        geom.box.first[dim]  = std::min(geom.box.first[dim], m_space.points[point_index * dimension + dim]);
                    }
                }
            }
        }
        return geom;
    }

    // Fills scratch.local_quadrature_points with the element's physical quadrature points; returns the jacobian.
    CoordinatePrecision map_element(int element_index, quadrature_scratch &scratch) const {
        for (int p = 0; p < m_space.number_of_points_per_element; p++) {
            int point_index = m_space.elements_to_points[element_index * m_space.number_of_points_per_element + p];
            for (int dim = 0; dim < dimension; dim++) {
                scratch.local_cell_points[p][dim] = m_space.points[point_index * dimension + dim];
            }
        }
        return scratch.map_cell_to_quadrature(m_setup);
    }

    // Local dof (in the range) of each of the element's dofs, -1 for those outside the range.
    std::vector<int> local_dofs(const side_geometry &geom, int element_index) const {
        std::vector<int> dofs(std::size_t(m_space.number_of_dofs_per_element), -1);
        for (int j = 0; j < m_space.number_of_dofs_per_element; j++) {
            auto it = geom.element_indices_to_dofs.find(std::make_pair(element_index, j));
            if (it != geom.element_indices_to_dofs.end())
                dofs[std::size_t(j)] = it->second;
        }
        return dofs;
    }

    // tmp_mat (n_interp x dofs per element) = S * phi for one element, where S holds the box's Lagrange
    // polynomials at the element's quadrature points and phi the weighted basis functions there.
    void compute_elementwise_interpolation_matrix(int L, int element_index, quadrature_scratch &scratch, const interpolation_box<CoordinatePrecision, dimension> &box, Matrix<CoefficientPrecision> &S, Matrix<CoefficientPrecision> &phi, Matrix<CoefficientPrecision> &tmp_mat) const {
        CoordinatePrecision jacobian = map_element(element_index, scratch);
        CoefficientPrecision *S_ptr  = S.data();
        theia::get_polynomials<dimension, CoordinatePrecision, CoefficientPrecision, 0>(L, S_ptr, box.first.data(), box.second.data(), scratch.local_quadrature_points.data(), scratch.local_quadrature_points.size());
        for (int i = 0; i < phi.nb_rows(); i++) {
            for (int j = 0; j < phi.nb_cols(); j++) {
                phi(i, j) = jacobian * m_setup.weights[i] * m_space.basis_function(element_index, j, m_setup.reference_quadrature_points[i]);
            }
        }
        add_matrix_matrix_product('N', 'N', CoefficientPrecision(1), S, phi, CoefficientPrecision(0), tmp_mat);
    }

  public:
    FEMDiscretization(FEMSpace<CoefficientPrecision, CoordinatePrecision, dimension> space, const Cluster<CoordinatePrecision> &cluster) : m_space(std::move(space)), m_cluster(&cluster), m_setup(m_space.quadrature_order, m_space.number_of_points_per_element) {}

    const Cluster<CoordinatePrecision> &cluster() const override { return *m_cluster; }

    interpolation_box<CoordinatePrecision, dimension> box(int offset, int size) const override {
        interpolation_box<CoordinatePrecision, dimension> result = collect_side(offset, size).box;
        inflate_box<CoordinatePrecision, dimension>(result);
        return result;
    }

    Matrix<CoefficientPrecision> interpolation_matrix(int L, interpolation_box<CoordinatePrecision, dimension> box, int offset, int size) const override {
        int n_interp       = number_of_interpolation_nodes<dimension>(L);
        side_geometry geom = collect_side(offset, size);
        Matrix<CoefficientPrecision> dof_to_interp(n_interp, size);
        quadrature_scratch scratch(m_space.number_of_points_per_element, m_setup.reference_quadrature_points.size());
        Matrix<CoefficientPrecision> tmp_mat(n_interp, m_space.number_of_dofs_per_element);
        Matrix<CoefficientPrecision> S(n_interp, scratch.local_quadrature_points.size());
        Matrix<CoefficientPrecision> phi(scratch.local_quadrature_points.size(), m_space.number_of_dofs_per_element);
        for (int element_index : geom.element_indices) {
            compute_elementwise_interpolation_matrix(L, element_index, scratch, box, S, phi, tmp_mat);
            std::vector<int> dofs = local_dofs(geom, element_index);
            for (int j = 0; j < m_space.number_of_dofs_per_element; j++) {
                if (dofs[std::size_t(j)] >= 0) {
                    for (int i = 0; i < n_interp; i++) {
                        dof_to_interp(i, dofs[std::size_t(j)]) += tmp_mat(i, j);
                    }
                }
            }
        }
        return dof_to_interp;
    }

    void tensor_interpolation(int L, interpolation_box<CoordinatePrecision, dimension> box, int offset, int size, theia::TensorP2M<dimension, CoefficientPrecision> &p2m, std::vector<interpolation_weight<CoefficientPrecision>> &w) const override {
        side_geometry geom = collect_side(offset, size);
        quadrature_scratch scratch(m_space.number_of_points_per_element, m_setup.reference_quadrature_points.size());
        std::vector<std::array<CoordinatePrecision, dimension>> particles;
        w.clear();
        for (int element_index : geom.element_indices) {
            CoordinatePrecision jacobian = map_element(element_index, scratch);
            std::vector<int> dofs        = local_dofs(geom, element_index);
            for (std::size_t q = 0; q < scratch.local_quadrature_points.size(); q++) {
                int particle = particles.size();
                particles.push_back(scratch.local_quadrature_points[q]);
                for (int j = 0; j < m_space.number_of_dofs_per_element; j++) {
                    if (dofs[std::size_t(j)] >= 0) {
                        w.push_back({particle, dofs[std::size_t(j)], jacobian * m_setup.weights[q] * m_space.basis_function(element_index, j, m_setup.reference_quadrature_points[q])});
                    }
                }
            }
        }
        std::array<int, dimension> Ls;
        Ls.fill(L);
        theia::P2M<dimension, CoordinatePrecision, CoefficientPrecision, 0>(box.first.data(), box.second.data(), int(particles.size()), particles.data(), Ls.data(), p2m, theia::op_tensor_interpolation<dimension, CoordinatePrecision, CoefficientPrecision, 0>{});
    }
};

} // namespace htool

#endif
