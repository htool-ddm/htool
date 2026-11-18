#ifndef HTOOL_TESTING_FEM_TEST_HPP
#define HTOOL_TESTING_FEM_TEST_HPP

#include "../matrix/matrix.hpp"
#include "../misc/fem_interface.hpp"
#include "../misc/misc.hpp"
#include "../quadratures/gauss_legendre.hpp"
#include "../quadratures/quadrature.hpp"
#include "../quadratures/triangle.hpp"
#include "mesh_test.hpp"
#include <array>
#include <cmath>
#include <complex>
#include <functional>
#include <map>
#include <stdexcept>
#include <vector>

namespace htool {

enum class ElementType { P0,
                         P1 };

// ComplexLaplace is (1 + i) / (4 pi r), the kernel of GeneratorTestComplex.
enum class KernelType { Constant,
                        Laplace,
                        Helmholtz,
                        ComplexLaplace };

template <typename CoefficientPrecision, int dimension>
using test_kernel = std::function<void(std::array<double, dimension> *, int, std::array<double, dimension> *, int, CoefficientPrecision *)>;

inline void kernel_value(KernelType kernel, double r, double &value) {
    value = kernel == KernelType::Constant ? 1. : 1. / (4. * M_PI * r);
}

inline void kernel_value(KernelType kernel, double r, std::complex<double> &value) {
    switch (kernel) {
    case KernelType::Constant:
        value = 1.;
        break;
    case KernelType::Laplace:
        value = 1. / (4. * M_PI * r);
        break;
    case KernelType::Helmholtz:
        value = std::exp(std::complex<double>(0., 1.) * r) / (4. * M_PI * r);
        break;
    case KernelType::ComplexLaplace:
        value = std::complex<double>(1., 1.) / (4. * M_PI * r);
        break;
    }
}

// Helmholtz and ComplexLaplace need a complex CoefficientPrecision.
template <typename CoefficientPrecision, int dimension>
test_kernel<CoefficientPrecision, dimension> make_kernel(KernelType kernel) {
    if (!is_complex<CoefficientPrecision>() && (kernel == KernelType::Helmholtz || kernel == KernelType::ComplexLaplace)) {
        throw std::invalid_argument("make_kernel: Helmholtz and ComplexLaplace kernels need a complex CoefficientPrecision");
    }
    return [kernel](std::array<double, dimension> *target_points, int Nx, std::array<double, dimension> *source_points, int Ny, CoefficientPrecision *mat) {
        for (std::size_t j = 0; j < std::size_t(Ny); j++) {
            for (std::size_t i = 0; i < std::size_t(Nx); i++) {
                double r2 = 0;
                for (std::size_t d = 0; d < std::size_t(dimension); d++)
                    r2 += (target_points[i][d] - source_points[j][d]) * (target_points[i][d] - source_points[j][d]);
                kernel_value(kernel, std::sqrt(r2), mat[i + j * std::size_t(Nx)]);
            }
        }
    };
}

// Finite element dofs of a TestMesh.
struct TestDofs {
    std::map<int, std::vector<int>> dofs_to_elements; // dof -> flattened (element, local dof) pairs, as FEMSpace expects
    std::vector<int> elements_to_dofs;                // element * number_of_dofs_per_element + local dof -> dof
    int number_of_dofs_per_element;
    std::vector<double> dof_points; // dimension coordinates per dof, for clustering: centroids (P0) or vertices (P1)

    int number_of_dofs() const { return int(dofs_to_elements.size()); }
};

template <int dimension>
TestDofs make_dofs(const TestMesh<dimension> &mesh, ElementType element) {
    TestDofs dofs;
    int k         = mesh.number_of_points_per_element;
    auto point_of = [&](int e, int p) { return mesh.elements_to_points[std::size_t(k * e + p)]; };
    if (element == ElementType::P0) {
        dofs.number_of_dofs_per_element = 1;
        for (int e = 0; e < mesh.number_of_elements(); e++) {
            dofs.dofs_to_elements[e] = {e, 0};
            dofs.elements_to_dofs.push_back(e);
            for (int d = 0; d < dimension; d++) {
                double centroid = 0;
                for (int p = 0; p < k; p++)
                    centroid += mesh.points[std::size_t(point_of(e, p) * dimension + d)];
                dofs.dof_points.push_back(centroid / k);
            }
        }
    } else {
        dofs.number_of_dofs_per_element = k;
        dofs.dof_points                 = mesh.points;
        for (int e = 0; e < mesh.number_of_elements(); e++) {
            for (int p = 0; p < k; p++) {
                int vertex = point_of(e, p);
                dofs.dofs_to_elements[vertex].push_back(e);
                dofs.dofs_to_elements[vertex].push_back(p);
                dofs.elements_to_dofs.push_back(vertex);
            }
        }
    }
    return dofs;
}

// P0 = 1. P1 on segments in the reference coordinate xi in [-1, 1]; on triangles in (xi, eta) of the unit triangle.
template <typename CoefficientPrecision, int dimension>
typename FEMSpace<CoefficientPrecision, double, dimension>::basis_function_type make_basis(ElementType element, int number_of_points_per_element) {
    if (element == ElementType::P0) {
        return [](int, int, const std::vector<double> &) { return CoefficientPrecision(1); };
    }
    if (number_of_points_per_element == 2) {
        return [](int, int local_dof, const std::vector<double> &x) {
            return CoefficientPrecision(local_dof == 0 ? 0.5 * (1. - x[0]) : 0.5 * (1. + x[0]));
        };
    }
    return [](int, int local_dof, const std::vector<double> &x) {
        return CoefficientPrecision(local_dof == 0 ? 1. - x[0] - x[1] : x[std::size_t(local_dof - 1)]);
    };
}

// mesh and dofs must outlive the returned FEMSpace.
template <typename CoefficientPrecision, int dimension>
FEMSpace<CoefficientPrecision, double, dimension> make_fem_space(const TestMesh<dimension> &mesh, const TestDofs &dofs, ElementType element, int quadrature_order) {
    return FEMSpace<CoefficientPrecision, double, dimension>{make_basis<CoefficientPrecision, dimension>(element, mesh.number_of_points_per_element), &dofs.dofs_to_elements, dofs.number_of_dofs_per_element, mesh.elements_to_points.data(), mesh.number_of_points_per_element, mesh.points.data(), mesh.points.size(), quadrature_order};
}

// A weighted point contributing to one dof: the integral of f against dof's basis function is the sum of
// weight * f(point) over its particles.
template <typename CoefficientPrecision, int dimension>
struct TestParticle {
    std::array<double, dimension> point;
    int dof;
    CoefficientPrecision weight;
};

// One particle per point, weight 1: the point cloud itself.
template <typename CoefficientPrecision, int dimension>
std::vector<TestParticle<CoefficientPrecision, dimension>> point_particles(const std::vector<double> &points) {
    std::vector<TestParticle<CoefficientPrecision, dimension>> particles(points.size() / std::size_t(dimension));
    for (std::size_t i = 0; i < particles.size(); i++) {
        for (std::size_t d = 0; d < std::size_t(dimension); d++)
            particles[i].point[d] = points[i * std::size_t(dimension) + d];
        particles[i].dof    = int(i);
        particles[i].weight = CoefficientPrecision(1);
    }
    return particles;
}

// Quadrature particles of a finite element space: for each element e, quadrature point q and local dof j, the
// point x_q with weight jacobian * w_q * phi_j(reference x_q). Uses the same quadrature rule as FEMDiscretization
// (find_best_rule) but its own affine maps and jacobians (measure / sum of weights), so it is independent of the
// code under test.
template <typename CoefficientPrecision, int dimension>
std::vector<TestParticle<CoefficientPrecision, dimension>> fem_particles(const TestMesh<dimension> &mesh, const TestDofs &dofs, ElementType element, int quadrature_order) {
    auto basis      = make_basis<CoefficientPrecision, dimension>(element, mesh.number_of_points_per_element);
    bool is_segment = mesh.number_of_points_per_element == 2;
    std::vector<std::vector<double>> reference_points;
    std::vector<double> weights;
    if (is_segment) {
        const auto &rule = find_best_rule(std::size_t(quadrature_order), gauss_legendre_rules<double>);
        for (std::size_t q = 0; q < rule.nb_points; q++) {
            reference_points.push_back({rule.quad_points[q].point[0]});
            weights.push_back(rule.quad_points[q].w);
        }
    } else {
        const auto &rule = find_best_rule(std::size_t(quadrature_order), triangle_rules<double>);
        for (std::size_t q = 0; q < rule.nb_points; q++) {
            reference_points.push_back({rule.quad_points[q].point[0], rule.quad_points[q].point[1]});
            weights.push_back(rule.quad_points[q].w);
        }
    }
    double weight_sum = 0;
    for (double w : weights)
        weight_sum += w;

    std::vector<TestParticle<CoefficientPrecision, dimension>> particles;
    int k = mesh.number_of_points_per_element;
    for (int e = 0; e < mesh.number_of_elements(); e++) {
        std::array<std::array<double, dimension>, 3> vertices{};
        for (int p = 0; p < k; p++)
            for (std::size_t d = 0; d < std::size_t(dimension); d++)
                vertices[std::size_t(p)][d] = mesh.points[std::size_t(mesh.elements_to_points[std::size_t(k * e + p)] * dimension) + d];
        std::array<double, dimension> e1{}, e2{};
        for (std::size_t d = 0; d < std::size_t(dimension); d++) {
            e1[d] = vertices[1][d] - vertices[0][d];
            e2[d] = vertices[2][d] - vertices[0][d];
        }
        double measure = 0;
        if (is_segment) {
            for (double x : e1)
                measure += x * x;
            measure = std::sqrt(measure);
        } else {
            // |e1 x e2| / 2, with the 2D vectors seen in the plane z = 0
            double cz = e1[0] * e2[1] - e1[1] * e2[0], cx = 0, cy = 0;
            if (dimension == 3) {
                cx = e1[1] * e2[dimension - 1] - e1[dimension - 1] * e2[1];
                cy = e1[dimension - 1] * e2[0] - e1[0] * e2[dimension - 1];
            }
            measure = std::sqrt(cx * cx + cy * cy + cz * cz) / 2;
        }
        double jacobian = measure / weight_sum;

        for (std::size_t q = 0; q < weights.size(); q++) {
            std::array<double, dimension> x = vertices[0];
            for (std::size_t d = 0; d < std::size_t(dimension); d++)
                x[d] += is_segment ? 0.5 * (1. + reference_points[q][0]) * e1[d] : reference_points[q][0] * e1[d] + reference_points[q][1] * e2[d];
            for (int j = 0; j < dofs.number_of_dofs_per_element; j++) {
                particles.push_back({x, dofs.elements_to_dofs[std::size_t(e * dofs.number_of_dofs_per_element + j)], CoefficientPrecision(jacobian * weights[q]) * basis(e, j, reference_points[q])});
            }
        }
    }
    return particles;
}

// Dense reference in user numbering: ref(i, j) = sum over particles p of dof i and q of dof j of
// w_p * w_q * kernel(x_p, y_q).
template <typename CoefficientPrecision, int dimension>
Matrix<CoefficientPrecision> reference_matrix(const test_kernel<CoefficientPrecision, dimension> &kernel, const std::vector<TestParticle<CoefficientPrecision, dimension>> &target, int number_of_target_dofs, const std::vector<TestParticle<CoefficientPrecision, dimension>> &source, int number_of_source_dofs) {
    Matrix<CoefficientPrecision> reference(number_of_target_dofs, number_of_source_dofs);
    std::vector<std::array<double, dimension>> target_points(target.size());
    for (std::size_t p = 0; p < target.size(); p++)
        target_points[p] = target[p].point;
    std::vector<CoefficientPrecision> column(target.size());
    for (const auto &particle : source) {
        std::array<double, dimension> y = particle.point;
        kernel(target_points.data(), int(target.size()), &y, 1, column.data());
        for (std::size_t p = 0; p < target.size(); p++)
            reference(target[p].dof, particle.dof) += target[p].weight * particle.weight * column[p];
    }
    return reference;
}

} // namespace htool

#endif
