#ifndef HTOOL_TESTS_LRMAT_BUILD_INTERPOLATION_HPP
#define HTOOL_TESTS_LRMAT_BUILD_INTERPOLATION_HPP

// Shared by the HCA tests: the four ways of computing a block, and a sweep checking them against a reference.

#include <array>
#include <atomic>
#include <htool/clustering/cluster_node.hpp>
#include <htool/clustering/tree_builder/tree_builder.hpp>
#include <htool/hmatrix/lrmat/interpolation.hpp>
#include <htool/hmatrix/lrmat/lrmat.hpp>
#include <htool/matrix/utils.hpp>
#include <htool/testing/fem_test.hpp>
#include <iostream>
#include <memory>
#include <vector>

namespace htool {

// Cluster tree on the dof points, leaves of at most 4 dofs.
inline Cluster<double> make_cluster(const TestDofs &dofs, int dimension) {
    ClusterTreeBuilder<double> builder;
    builder.set_maximal_leaf_size(4);
    return builder.create_cluster_tree(dofs.number_of_dofs(), dimension, dofs.dof_points.data(), 2, 2);
}

inline bool is_identity(const std::vector<int> &permutation) {
    for (std::size_t i = 0; i < permutation.size(); i++) {
        if (permutation[i] != int(i))
            return false;
    }
    return true;
}

// Block {row_offset, M} x {col_offset, N} in tree numbering of a matrix in user numbering.
template <typename CoefficientPrecision>
Matrix<CoefficientPrecision> permuted_block(const Matrix<CoefficientPrecision> &reference, const Cluster<double> &target_cluster, const Cluster<double> &source_cluster, int row_offset, int M, int col_offset, int N) {
    Matrix<CoefficientPrecision> block(M, N);
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < M; i++) {
            block(i, j) = reference(target_cluster.get_permutation()[row_offset + i], source_cluster.get_permutation()[col_offset + j]);
        }
    }
    return block;
}

// Forwards to another discretization and counts its calls: leaf_calls (interpolation_matrix, tensor_interpolation)
// and calls (all three methods).
template <typename CoefficientPrecision, int dimension>
class CountingDiscretization final : public VirtualDiscretization<CoefficientPrecision, double, dimension> {
    std::shared_ptr<const VirtualDiscretization<CoefficientPrecision, double, dimension>> m_inner;

  public:
    mutable std::atomic<int> leaf_calls{0}, calls{0};
    explicit CountingDiscretization(std::shared_ptr<const VirtualDiscretization<CoefficientPrecision, double, dimension>> inner) : m_inner(std::move(inner)) {}
    const Cluster<double> &cluster() const override { return m_inner->cluster(); }
    interpolation_box<double, dimension> box(int offset, int size) const override {
        calls++;
        return m_inner->box(offset, size);
    }
    Matrix<CoefficientPrecision> interpolation_matrix(int L, interpolation_box<double, dimension> box, int offset, int size) const override {
        calls++;
        leaf_calls++;
        return m_inner->interpolation_matrix(L, box, offset, size);
    }
    void tensor_interpolation(int L, interpolation_box<double, dimension> box, int offset, int size, theia::TensorP2M<dimension, CoefficientPrecision> &p2m, std::vector<interpolation_weight<CoefficientPrecision>> &w) const override {
        calls++;
        leaf_calls++;
        m_inner->tensor_interpolation(L, box, offset, size, p2m, w);
    }
};

// The four ways HCA computes a block: the direct path, and the three reinterpolation modes of the precomputed trees.
// Each compressor sees the discretizations through its own counting wrappers, so that a test can tell which blocks
// went through the discretizations (direct path) and which only through the precomputed trees.
template <typename CoefficientPrecision, int dimension>
struct HCACompressors {
    using compressor_type     = HCA<CoefficientPrecision, double, dimension>;
    using discretization_type = typename compressor_type::discretization_type;
    using counting_type       = CountingDiscretization<CoefficientPrecision, dimension>;
    using Mode                = typename compressor_type::ReinterpolationMode;

    std::array<std::shared_ptr<counting_type>, 4> targets, sources;
    std::array<std::unique_ptr<compressor_type>, 4> compressors;

    HCACompressors(typename compressor_type::kernel_type kernel, const std::shared_ptr<const discretization_type> &target, const std::shared_ptr<const discretization_type> &source, double epsilon) {
        for (int way = 0; way < 4; way++) {
            targets[way] = std::make_shared<counting_type>(target);
            sources[way] = target == source ? targets[way] : std::make_shared<counting_type>(source); // keeps the shared precomputation
        }
        compressors[0] = std::unique_ptr<compressor_type>(new compressor_type(kernel, targets[0], sources[0]));
        compressors[1] = std::unique_ptr<compressor_type>(new compressor_type(kernel, targets[1], sources[1], epsilon, Mode::InterpolationMatrices));
        compressors[2] = std::unique_ptr<compressor_type>(new compressor_type(kernel, targets[2], sources[2], epsilon, Mode::LowRankFactors));
        compressors[3] = std::unique_ptr<compressor_type>(new compressor_type(kernel, targets[3], sources[3], epsilon, Mode::TensorLowRankFactors));
    }

    std::array<const compressor_type *, 4> all() const { return {compressors[0].get(), compressors[1].get(), compressors[2].get(), compressors[3].get()}; }

    // Calls to way's discretizations so far.
    int calls(int way) const { return targets[way]->calls + (sources[way] == targets[way] ? 0 : int(sources[way]->calls)); }
};

// Root, then one node per level down the first branch, ending at a leaf.
inline std::vector<const Cluster<double> *> first_branch(const Cluster<double> &root) {
    std::vector<const Cluster<double> *> nodes{&root};
    const Cluster<double> *node = &root;
    while (!node->is_leaf()) {
        node = node->get_children().front().get();
        nodes.push_back(node);
    }
    return nodes;
}

// reference is the whole target x source matrix in user numbering. Checks, for each of the four ways:
// - every pair of first_branch(target) x first_branch(source), compressed in parallel, and a fixed rank of 5 on the
//   root block: these are tree nodes, so the precomputed modes must not call their discretizations;
// - a block that is not a tree node, {0, M - 1} x {0, N}: the precomputed modes must fall back to the direct path,
//   i.e. call their discretizations.
// Every block must be computed with relative Frobenius error < epsilon * (1 + tolerance), and the three precomputed
// modes must agree with each other to 1e-10 (the direct path uses other boxes for internal nodes, so it only meets
// epsilon). Returns true on error.
template <typename CoefficientPrecision, int dimension>
bool check_hca_modes(const HCACompressors<CoefficientPrecision, dimension> &hca, const Cluster<double> &target_cluster, const Cluster<double> &source_cluster, const Matrix<CoefficientPrecision> &reference, double epsilon, double tolerance) {
    auto compressors = hca.all();
    std::vector<std::array<int, 4>> blocks; // {row offset, M, col offset, N}
    for (const Cluster<double> *t : first_branch(target_cluster)) {
        for (const Cluster<double> *s : first_branch(source_cluster)) {
            blocks.push_back({t->get_offset(), t->get_size(), s->get_offset(), s->get_size()});
        }
    }
    blocks.push_back({0, target_cluster.get_size() - 1, 0, source_cluster.get_size()}); // not a tree node: last
    const std::size_t number_of_tree_blocks = blocks.size() - 1;

    std::vector<std::array<double, 4>> errors(blocks.size()), differences(blocks.size());
    std::vector<std::array<int, 4>> computed(blocks.size());
    auto compute_block = [&](std::size_t b) {
        int row_offset = blocks[b][0], M = blocks[b][1], col_offset = blocks[b][2], N = blocks[b][3];
        Matrix<CoefficientPrecision> ref = permuted_block(reference, target_cluster, source_cluster, row_offset, M, col_offset, N), first_mode;
        for (int way = 0; way < 4; way++) {
            LowRankMatrix<CoefficientPrecision> lrmat(M, N, epsilon);
            computed[b][way] = compressors[way]->copy_low_rank_approximation(M, N, row_offset, col_offset, lrmat);
            Matrix<CoefficientPrecision> dense(M, N);
            if (computed[b][way])
                lrmat.copy_to_dense(dense.data());
            errors[b][way] = normFrob(ref - dense) / normFrob(ref);
            if (way == 1)
                first_mode = dense;
            differences[b][way] = way < 1 ? 0 : normFrob(first_mode - dense) / normFrob(first_mode);
        }
    };

    bool is_error = false;
    std::array<int, 4> calls_after_construction{hca.calls(0), hca.calls(1), hca.calls(2), hca.calls(3)};
#if defined(_OPENMP)
#    pragma omp parallel for
#endif
    for (int b = 0; b < int(number_of_tree_blocks); b++) {
        compute_block(std::size_t(b));
    }
    for (const auto *compressor : compressors) {
        int reqrank = 5;
        LowRankMatrix<CoefficientPrecision> lrmat(target_cluster.get_size(), source_cluster.get_size(), epsilon);
        bool fixed_computed = compressor->copy_low_rank_approximation(target_cluster.get_size(), source_cluster.get_size(), 0, 0, reqrank, lrmat);
        is_error            = is_error || !fixed_computed || lrmat.get_U().nb_cols() != reqrank || lrmat.get_V().nb_rows() != reqrank;
    }
    std::array<int, 4> calls_after_tree_blocks{hca.calls(0), hca.calls(1), hca.calls(2), hca.calls(3)};
    compute_block(number_of_tree_blocks);
    std::array<int, 4> calls_after_fallback{hca.calls(0), hca.calls(1), hca.calls(2), hca.calls(3)};

    for (std::size_t b = 0; b < blocks.size(); b++) {
        std::cout << "  block " << blocks[b][1] << "x" << blocks[b][3] << " at (" << blocks[b][0] << ", " << blocks[b][2] << ") errors:";
        for (int way = 0; way < 4; way++)
            std::cout << ' ' << (computed[b][way] ? errors[b][way] : -1.);
        std::cout << ", mode differences: " << differences[b][2] << ' ' << differences[b][3] << '\n';
        for (int way = 0; way < 4; way++) {
            is_error = is_error || !computed[b][way] || !(errors[b][way] < epsilon * (1 + tolerance)) || !(differences[b][way] < 1e-10);
        }
    }
    std::cout << "  discretization calls on tree blocks:";
    for (int way = 0; way < 4; way++)
        std::cout << ' ' << calls_after_tree_blocks[way] - calls_after_construction[way];
    std::cout << ", on the fallback block:";
    for (int way = 0; way < 4; way++)
        std::cout << ' ' << calls_after_fallback[way] - calls_after_tree_blocks[way];
    std::cout << '\n';
    for (int way = 1; way < 4; way++) {
        is_error = is_error || calls_after_tree_blocks[way] != calls_after_construction[way] || calls_after_fallback[way] == calls_after_tree_blocks[way];
    }
    if (is_error)
        std::cout << "  FAILED\n";
    return is_error;
}

} // namespace htool

#endif
