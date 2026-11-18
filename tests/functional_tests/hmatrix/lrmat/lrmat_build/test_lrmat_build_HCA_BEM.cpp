// HCA for boundary element matrices: finite element spaces (P0/P1 on triangles and segments, 2D and 3D) and
// potentials (points x finite element space), in the four ways of computing a block.

#include "test_lrmat_build_interpolation.hpp"
#include <htool/clustering/tree_builder/tree_builder.hpp>
#include <htool/hmatrix/lrmat/interpolation.hpp>
#include <htool/testing/fem_test.hpp>
#include <iostream>
#include <memory>
#include <string>

using namespace std;
using namespace htool;

constexpr int quadrature_order = 10;
constexpr double tolerance     = 1.;

// Constant kernel: the reference integrates 1, so P0 entries are products of element measures and P1 entries sum to
// the product of the total measures (partition of unity).
template <int dimension>
bool test_reference_exact_values(const TestMesh<dimension> &mesh, double element_measure, double total_measure, const string &name) {
    auto kernel               = make_kernel<double, dimension>(KernelType::Constant);
    TestDofs p0               = make_dofs(mesh, ElementType::P0);
    TestDofs p1               = make_dofs(mesh, ElementType::P1);
    auto particles0           = fem_particles<double, dimension>(mesh, p0, ElementType::P0, quadrature_order);
    auto particles1           = fem_particles<double, dimension>(mesh, p1, ElementType::P1, quadrature_order);
    Matrix<double> reference0 = reference_matrix<double, dimension>(kernel, particles0, p0.number_of_dofs(), particles0, p0.number_of_dofs());
    Matrix<double> reference1 = reference_matrix<double, dimension>(kernel, particles1, p1.number_of_dofs(), particles1, p1.number_of_dofs());
    double p0_error = 0, p1_sum = 0;
    for (int j = 0; j < reference0.nb_cols(); j++)
        for (int i = 0; i < reference0.nb_rows(); i++)
            p0_error = std::max(p0_error, std::abs(reference0(i, j) - element_measure * element_measure));
    for (int j = 0; j < reference1.nb_cols(); j++)
        for (int i = 0; i < reference1.nb_rows(); i++)
            p1_sum += reference1(i, j);
    cout << name << ": P0 max error " << p0_error << ", P1 sum " << p1_sum << " (expected " << total_measure * total_measure << ")\n";
    return !(p0_error < 1e-12) || !(std::abs(p1_sum - total_measure * total_measure) < 1e-12);
}

template <typename CoefficientPrecision, int dimension>
bool test_fem(const TestMesh<dimension> &target_mesh, const TestMesh<dimension> &source_mesh, ElementType element, KernelType kernel_type, double epsilon, const string &name) {
    cout << epsilon << ' ' << name << '\n';
    auto kernel                      = make_kernel<CoefficientPrecision, dimension>(kernel_type);
    TestDofs target_dofs             = make_dofs(target_mesh, element);
    TestDofs source_dofs             = make_dofs(source_mesh, element);
    Cluster<double> target_cluster   = make_cluster(target_dofs, dimension);
    Cluster<double> source_cluster   = make_cluster(source_dofs, dimension);
    bool non_trivial_numbering_error = is_identity(target_cluster.get_permutation()) || is_identity(source_cluster.get_permutation()); // tree vs user numbering must be exercised
    if (non_trivial_numbering_error)
        cout << "  FAILED: identity cluster permutation\n";
    auto target = std::make_shared<FEMDiscretization<CoefficientPrecision, double, dimension>>(make_fem_space<CoefficientPrecision, dimension>(target_mesh, target_dofs, element, quadrature_order), target_cluster);
    auto source = std::make_shared<FEMDiscretization<CoefficientPrecision, double, dimension>>(make_fem_space<CoefficientPrecision, dimension>(source_mesh, source_dofs, element, quadrature_order), source_cluster);
    HCACompressors<CoefficientPrecision, dimension> compressors(kernel, target, source, epsilon);

    Matrix<CoefficientPrecision> reference = reference_matrix<CoefficientPrecision, dimension>(kernel, fem_particles<CoefficientPrecision, dimension>(target_mesh, target_dofs, element, quadrature_order), target_dofs.number_of_dofs(), fem_particles<CoefficientPrecision, dimension>(source_mesh, source_dofs, element, quadrature_order), source_dofs.number_of_dofs());
    return check_hca_modes<CoefficientPrecision, dimension>(compressors, target_cluster, source_cluster, reference, epsilon, tolerance) || non_trivial_numbering_error;
}

// Potential: point targets (the centroids of target_mesh), finite element sources.
template <typename CoefficientPrecision>
bool test_potential(ElementType element, KernelType kernel_type, double epsilon, const string &name) {
    constexpr int dimension = 3;
    cout << epsilon << ' ' << name << '\n';
    auto kernel                     = make_kernel<CoefficientPrecision, dimension>(kernel_type);
    TestMesh<dimension> target_mesh = make_triangle_mesh<dimension>(4, 0), source_mesh = make_triangle_mesh<dimension>(4, 2);
    TestDofs targets                 = make_dofs(target_mesh, ElementType::P0);
    TestDofs source_dofs             = make_dofs(source_mesh, element);
    Cluster<double> target_cluster   = make_cluster(targets, dimension);
    Cluster<double> source_cluster   = make_cluster(source_dofs, dimension);
    bool non_trivial_numbering_error = is_identity(target_cluster.get_permutation()) || is_identity(source_cluster.get_permutation());
    auto target                      = std::make_shared<PointDiscretization<CoefficientPrecision, double, dimension>>(targets.dof_points.data(), target_cluster);
    auto source                      = std::make_shared<FEMDiscretization<CoefficientPrecision, double, dimension>>(make_fem_space<CoefficientPrecision, dimension>(source_mesh, source_dofs, element, quadrature_order), source_cluster);
    HCACompressors<CoefficientPrecision, dimension> compressors(kernel, target, source, epsilon);

    Matrix<CoefficientPrecision> reference = reference_matrix<CoefficientPrecision, dimension>(kernel, point_particles<CoefficientPrecision, dimension>(targets.dof_points), targets.number_of_dofs(), fem_particles<CoefficientPrecision, dimension>(source_mesh, source_dofs, element, quadrature_order), source_dofs.number_of_dofs());
    return check_hca_modes<CoefficientPrecision, dimension>(compressors, target_cluster, source_cluster, reference, epsilon, tolerance) || non_trivial_numbering_error;
}

// Every way of computing a block, for one element shape and dimension: P0/P1 x the three kernels.
template <int dimension>
bool test_element_shape(const TestMesh<dimension> &target_mesh, const TestMesh<dimension> &source_mesh, double epsilon, const string &name) {
    bool is_error = false;
    for (ElementType element : {ElementType::P0, ElementType::P1}) {
        string element_name = element == ElementType::P0 ? " P0" : " P1";
        is_error            = test_fem<double, dimension>(target_mesh, source_mesh, element, KernelType::Constant, epsilon, name + element_name + " Constant") || is_error;
        is_error            = test_fem<double, dimension>(target_mesh, source_mesh, element, KernelType::Laplace, epsilon, name + element_name + " Laplace") || is_error;
        is_error            = test_fem<std::complex<double>, dimension>(target_mesh, source_mesh, element, KernelType::Helmholtz, epsilon, name + element_name + " Helmholtz") || is_error;
    }
    return is_error;
}

int main(int, char *[]) {
    bool is_error = false;

    is_error = test_reference_exact_values<3>(make_triangle_mesh<3>(4, 0), 1. / 32, 1., "reference triangles 3D") || is_error;
    is_error = test_reference_exact_values<2>(make_triangle_mesh<2>(4, 0), 1. / 32, 1., "reference triangles 2D") || is_error;
    is_error = test_reference_exact_values<3>(make_segment_mesh<3>(16, 0), 1. / 16, 1., "reference segments 3D") || is_error;
    is_error = test_reference_exact_values<2>(make_segment_mesh<2>(16, 0), 1. / 16, 1., "reference segments 2D") || is_error;

    // The sweep must not pass when nothing can be compressed: target and source on the same mesh.
    cout << "vacuous sweep (expected to fail)\n";
    bool vacuous_is_error = test_fem<double, 3>(make_triangle_mesh<3>(4, 0), make_triangle_mesh<3>(4, 0), ElementType::P0, KernelType::Laplace, 1e-4, "same mesh");
    is_error              = !vacuous_is_error || is_error;

    double epsilon = 1e-4;
    is_error       = test_element_shape<3>(make_triangle_mesh<3>(4, 0), make_triangle_mesh<3>(4, 2), epsilon, "triangles 3D") || is_error;
    is_error       = test_element_shape<2>(make_triangle_mesh<2>(4, 0), make_triangle_mesh<2>(4, 3), epsilon, "triangles 2D") || is_error;
    is_error       = test_element_shape<3>(make_segment_mesh<3>(16, 0), make_segment_mesh<3>(16, 2), epsilon, "segments 3D") || is_error;
    is_error       = test_element_shape<2>(make_segment_mesh<2>(16, 0), make_segment_mesh<2>(16, 2), epsilon, "segments 2D") || is_error;
    is_error       = test_potential<double>(ElementType::P0, KernelType::Laplace, epsilon, "potential P0 Laplace") || is_error;
    is_error       = test_potential<std::complex<double>>(ElementType::P1, KernelType::Helmholtz, epsilon, "potential P1 Helmholtz") || is_error;

    // High precision on representative real cases only: each block then needs a dense 729 x 729 SVD in 3D
    // (complex cases at 1e-8 cost about three times more and are covered at 1e-4).
    epsilon  = 1e-8;
    is_error = test_fem<double, 3>(make_triangle_mesh<3>(4, 0), make_triangle_mesh<3>(4, 2), ElementType::P1, KernelType::Laplace, epsilon, "triangles 3D P1 Laplace") || is_error;
    is_error = test_potential<double>(ElementType::P0, KernelType::Laplace, epsilon, "potential P0 Laplace") || is_error;

    return is_error;
}
