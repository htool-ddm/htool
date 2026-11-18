// HCA on point clouds, in the four ways of computing a block (direct path and the three reinterpolation modes):
// test_lrmat as for the other compression strategies, a sweep comparing the four ways on tree blocks, the shared
// precomputation when target == source, and the fallback when a block asks for another epsilon than the trees.

#include "../test_lrmat_build.hpp" // for test_lrmat
#include "test_lrmat_build_interpolation.hpp"
#include <algorithm>
#include <htool/clustering/cluster_node.hpp>
#include <htool/clustering/tree_builder/tree_builder.hpp>
#include <htool/hmatrix/interfaces/virtual_generator.hpp>
#include <htool/hmatrix/lrmat/interpolation.hpp>
#include <htool/hmatrix/lrmat/lrmat.hpp>
#include <htool/testing/fem_test.hpp>
#include <htool/testing/generator_test.hpp>
#include <htool/testing/geometry.hpp>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

using namespace std;
using namespace htool;

constexpr int dimension = 3;

// Spheres of 500 targets and 100 sources at distances 15 to 40, as the other compression strategies' tests;
// root x root block in each of the four ways, fixed rank 10 and automatic rank, checked by test_lrmat.
template <typename CoefficientPrecision, typename Generator>
bool test_spheres(KernelType kernel_type, std::pair<double, double> fixed_compression_interval, std::pair<double, double> auto_compression_interval) {
    const double epsilon = 1e-4;
    const int nr = 500, nc = 100, reqrank_max = 10;
    std::vector<double> xt(3 * nr), xs(3 * nc);
    auto kernel   = make_kernel<CoefficientPrecision, dimension>(kernel_type);
    bool is_error = false;
    for (double distance : {15., 20., 30., 40.}) {
        create_sphere(nr, xt.data());
        create_sphere(nc, xs.data(), {0, 0, distance});
        ClusterTreeBuilder<double> builder;
        Cluster<double> t = builder.create_cluster_tree(nr, 3, xt.data(), 2, 2);
        Cluster<double> s = builder.create_cluster_tree(nc, 3, xs.data(), 2, 2);

        Generator A_in_user_numbering(3, xt, xs);
        InternalGeneratorWithPermutation<CoefficientPrecision> A(A_in_user_numbering, t.get_permutation().data(), s.get_permutation().data());

        HCACompressors<CoefficientPrecision, dimension> compressors(kernel, std::make_shared<PointDiscretization<CoefficientPrecision, double, dimension>>(xt.data(), t), std::make_shared<PointDiscretization<CoefficientPrecision, double, dimension>>(xs.data(), s), epsilon);
        for (const auto *compressor : compressors.all()) {
            LowRankMatrix<CoefficientPrecision> fixed(t.get_size(), s.get_size(), reqrank_max, epsilon);
            compressor->copy_low_rank_approximation(t.get_size(), s.get_size(), t.get_offset(), s.get_offset(), reqrank_max, fixed);
            LowRankMatrix<CoefficientPrecision> automatic(t.get_size(), s.get_size(), epsilon);
            compressor->copy_low_rank_approximation(t.get_size(), s.get_size(), t.get_offset(), s.get_offset(), automatic);
            cout << "distance " << distance << ": space saving fixed " << fixed.space_saving() << ", automatic " << automatic.space_saving() << '\n';
            is_error = test_lrmat<CoefficientPrecision>(t, s, A, fixed, automatic, fixed_compression_interval, auto_compression_interval, reqrank_max, epsilon) || is_error;
        }
    }
    return is_error;
}

// Centroids of two triangle meshes; tree blocks checked against the kernel at the points.
template <typename CoefficientPrecision>
bool test_point_sweep(KernelType kernel_type, double epsilon) {
    cout << epsilon << " point sweep\n";
    auto kernel                    = make_kernel<CoefficientPrecision, dimension>(kernel_type);
    TestDofs targets               = make_dofs(make_triangle_mesh<dimension>(4, 0), ElementType::P0);
    TestDofs sources               = make_dofs(make_triangle_mesh<dimension>(4, 10), ElementType::P0);
    Cluster<double> target_cluster = make_cluster(targets, dimension);
    Cluster<double> source_cluster = make_cluster(sources, dimension);
    HCACompressors<CoefficientPrecision, dimension> compressors(kernel, std::make_shared<PointDiscretization<CoefficientPrecision, double, dimension>>(targets.dof_points.data(), target_cluster), std::make_shared<PointDiscretization<CoefficientPrecision, double, dimension>>(sources.dof_points.data(), source_cluster), epsilon);
    Matrix<CoefficientPrecision> reference = reference_matrix<CoefficientPrecision, dimension>(kernel, point_particles<CoefficientPrecision, dimension>(targets.dof_points), targets.number_of_dofs(), point_particles<CoefficientPrecision, dimension>(sources.dof_points), sources.number_of_dofs());
    return check_hca_modes<CoefficientPrecision, dimension>(compressors, target_cluster, source_cluster, reference, epsilon, 1.);
}

// Same discretization on both sides builds each leaf once; two discretizations build them twice. Both compressors
// return the same blocks.
bool test_symmetric_sharing(double epsilon) {
    using Compressor = HCA<double, double, dimension>;
    cout << "symmetric sharing\n";
    auto kernel              = make_kernel<double, dimension>(KernelType::Laplace);
    TestMesh<dimension> mesh = make_triangle_mesh<dimension>(4, 0);
    TestDofs dofs            = make_dofs(mesh, ElementType::P0);
    Cluster<double> cluster  = make_cluster(dofs, dimension);
    auto points              = std::make_shared<PointDiscretization<double, double, dimension>>(dofs.dof_points.data(), cluster);

    int number_of_leaves = 0;
    std::vector<const Cluster<double> *> nodes;
    preorder_tree_traversal(cluster, [&](const Cluster<double> &node) {
        nodes.push_back(&node);
        number_of_leaves += node.is_leaf() ? 1 : 0;
    });

    bool is_error = false;
    for (auto mode : {Compressor::ReinterpolationMode::InterpolationMatrices, Compressor::ReinterpolationMode::LowRankFactors, Compressor::ReinterpolationMode::TensorLowRankFactors}) {
        auto shared = std::make_shared<CountingDiscretization<double, dimension>>(points);
        Compressor shared_compressor(kernel, shared, shared, epsilon, mode);
        auto target = std::make_shared<CountingDiscretization<double, dimension>>(points), source = std::make_shared<CountingDiscretization<double, dimension>>(points);
        Compressor separate_compressor(kernel, target, source, epsilon, mode);
        cout << "  leaves " << number_of_leaves << ", leaf constructions: shared " << shared->leaf_calls << ", separate " << target->leaf_calls + source->leaf_calls << '\n';
        is_error = is_error || shared->leaf_calls != number_of_leaves || target->leaf_calls + source->leaf_calls != 2 * number_of_leaves;

        int number_of_compressed_blocks = 0;
        double max_difference           = 0;
        for (const Cluster<double> *t : nodes) {
            for (const Cluster<double> *s : nodes) {
                LowRankMatrix<double> shared_lrmat(t->get_size(), s->get_size(), epsilon), separate_lrmat(t->get_size(), s->get_size(), epsilon);
                bool shared_computed   = shared_compressor.copy_low_rank_approximation(t->get_size(), s->get_size(), t->get_offset(), s->get_offset(), shared_lrmat);
                bool separate_computed = separate_compressor.copy_low_rank_approximation(t->get_size(), s->get_size(), t->get_offset(), s->get_offset(), separate_lrmat);
                is_error               = is_error || shared_computed != separate_computed;
                if (shared_computed && separate_computed) {
                    number_of_compressed_blocks++;
                    Matrix<double> shared_dense(t->get_size(), s->get_size()), separate_dense(t->get_size(), s->get_size());
                    shared_lrmat.copy_to_dense(shared_dense.data());
                    separate_lrmat.copy_to_dense(separate_dense.data());
                    max_difference = std::max(max_difference, normFrob(shared_dense - separate_dense) / normFrob(separate_dense));
                }
            }
        }
        cout << "  compressed blocks " << number_of_compressed_blocks << " of " << nodes.size() * nodes.size() << ", max shared/separate difference " << max_difference << '\n';
        is_error = is_error || number_of_compressed_blocks == 0 || !(max_difference < 1e-12);
    }
    return is_error;
}

// A block asking for another epsilon than the trees were precomputed with must meet its own epsilon: a tighter one
// falls back to the direct path (calls the discretizations), a looser one reuses the trees (no calls).
bool test_epsilon_mismatch() {
    const double tree_epsilon = 1e-4;
    cout << "epsilon mismatch\n";
    auto kernel                    = make_kernel<double, dimension>(KernelType::Laplace);
    TestDofs targets               = make_dofs(make_triangle_mesh<dimension>(4, 0), ElementType::P0);
    TestDofs sources               = make_dofs(make_triangle_mesh<dimension>(4, 3), ElementType::P0); // close enough that order 5 (1e-4) misses 1e-8
    Cluster<double> target_cluster = make_cluster(targets, dimension);
    Cluster<double> source_cluster = make_cluster(sources, dimension);
    auto target                    = std::make_shared<CountingDiscretization<double, dimension>>(std::make_shared<PointDiscretization<double, double, dimension>>(targets.dof_points.data(), target_cluster));
    auto source                    = std::make_shared<CountingDiscretization<double, dimension>>(std::make_shared<PointDiscretization<double, double, dimension>>(sources.dof_points.data(), source_cluster));
    HCA<double, double, dimension> compressor(kernel, target, source, tree_epsilon);

    int M = target_cluster.get_size(), N = source_cluster.get_size();
    Matrix<double> block = permuted_block(reference_matrix<double, dimension>(kernel, point_particles<double, dimension>(targets.dof_points), M, point_particles<double, dimension>(sources.dof_points), N), target_cluster, source_cluster, 0, M, 0, N);
    bool is_error        = false;
    for (double block_epsilon : {1e-8, 1e-2}) {
        int calls_before = target->calls + source->calls;
        LowRankMatrix<double> lrmat(M, N, block_epsilon);
        bool computed = compressor.copy_low_rank_approximation(M, N, 0, 0, lrmat);
        bool direct   = target->calls + source->calls != calls_before;
        Matrix<double> dense(M, N);
        lrmat.copy_to_dense(dense.data());
        double error = normFrob(block - dense) / normFrob(block);
        cout << "  tree epsilon " << tree_epsilon << ", block epsilon " << block_epsilon << ": error " << error << (direct ? ", direct path\n" : ", trees\n");
        is_error = is_error || !computed || !(error < 2 * block_epsilon) || direct != (block_epsilon < tree_epsilon);
    }
    return is_error;
}

int main(int, char *[]) {
    bool is_error = false;
    is_error      = test_spheres<double, GeneratorTestDouble>(KernelType::Laplace, {0.87, 0.89}, {0.93, 0.96}) || is_error;
    is_error      = test_spheres<std::complex<double>, GeneratorTestComplex>(KernelType::ComplexLaplace, {0.87, 0.89}, {0.93, 0.96}) || is_error;
    is_error      = test_point_sweep<double>(KernelType::Laplace, 1e-4) || is_error;
    is_error      = test_point_sweep<std::complex<double>>(KernelType::Helmholtz, 1e-4) || is_error;
    is_error      = test_point_sweep<double>(KernelType::Laplace, 1e-8) || is_error;
    is_error      = test_symmetric_sharing(1e-4) || is_error;
    is_error      = test_epsilon_mismatch() || is_error;
    cout << "test : " << is_error << '\n';
    return is_error;
}
