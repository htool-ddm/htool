#ifndef HTOOL_VIRTUAL_IBCOMP_HPP
#define HTOOL_VIRTUAL_IBCOMP_HPP

#include "../../hmatrix/interfaces/virtual_lrmat_generator.hpp" // for Virt...
#include "htool/basic_types/tree.hpp"
#include "htool/clustering/cluster_node.hpp"
#include "htool/matrix/linalg/add_matrix_matrix_product.hpp"
#include "htool/matrix/linalg/transpose.hpp"
#include "htool/matrix/utils/SVD_truncation.hpp"
#include "htool/misc/logger.hpp"
#include "htool/misc/misc.hpp"
#include "htool/wrappers/wrapper_blas.hpp"
#include "interpolation/discretization.hpp"
#include "interpolation/fem_discretization.hpp"
#include "interpolation/point_discretization.hpp"
#include <algorithm>
#include <functional>
#include <map>
#include <memory>
#include <sstream>
#include <utility>
#include <vector>

namespace htool {

// Low-rank approximation by Chebyshev interpolation of the kernel between the boxes of a block. The target
// and source sides are each described by a VirtualDiscretization (point cloud, finite element space, ...).
template <typename CoefficientPrecision, typename CoordinatePrecision, int dimension>
class HCA final : public VirtualInternalLowRankGenerator<CoefficientPrecision> {
  public:
    typedef const std::function<void(std::array<CoordinatePrecision, dimension> *, int, std::array<CoordinatePrecision, dimension> *, int, CoefficientPrecision *)> kernel_type;
    using discretization_type = VirtualDiscretization<CoefficientPrecision, CoordinatePrecision, dimension>;

    // How tree-node blocks are reinterpolated from the leaves (cluster-tree constructor only):
    // InterpolationMatrices rebuilds the block's n_interp x size interpolation matrices, then multiplies them
    // by the truncated SVD factors; LowRankFactors pushes the n_interp x rank SVD factors down to the leaves instead;
    // TensorLowRankFactors stores each node's P2M at its subtree's particles in tensor form and applies it,
    // with the leaves' weights W, directly to the SVD factors.
    enum class ReinterpolationMode { InterpolationMatrices,
                                     LowRankFactors,
                                     TensorLowRankFactors };

    bool check_size = false;

  private:
    using box_type = interpolation_box<CoordinatePrecision, dimension>;
    using node_key = std::pair<int, int>; // {offset, size}
    struct node_data {
        box_type box;                                                                            // inflated
        Matrix<CoefficientPrecision> leaf_matrix;                                                // leaves: n_interp x size, internal nodes: empty
        std::vector<std::pair<node_key, theia::Kron<dimension, CoefficientPrecision>>> children; // internal nodes: M2M child -> this
        int height = 0;                                                                          // levels of stored nodes below this one (0 at leaves)
        // TensorLowRankFactors only:
        theia::TensorP2M<dimension, CoefficientPrecision> p2m;     // P2M on this box at the subtree's particles
        std::vector<interpolation_weight<CoefficientPrecision>> w; // leaves: particle -> leaf dof
    };

    kernel_type m_kernel;
    std::shared_ptr<const discretization_type> m_target;
    std::shared_ptr<const discretization_type> m_source;
    // mutable: theia's gemm takes Kron by non-const reference; applies only read once prcmp() has run.
    mutable std::map<node_key, node_data> m_target_nodes, m_source_nodes;
    bool m_symmetric = false; // same discretization on both sides: m_source_nodes stays empty

    std::map<node_key, node_data> &source_nodes() const { return m_symmetric ? m_target_nodes : m_source_nodes; }
    underlying_type<CoefficientPrecision> m_tree_epsilon = 0;
    int m_tree_L                                         = 0;
    int m_tree_n_interp                                  = 0;
    ReinterpolationMode m_tree_mode                      = ReinterpolationMode::InterpolationMatrices;

    static bool boxes_intersect(const box_type &a, const box_type &b) {
        for (int dim = 0; dim < dimension; dim++) {
            if (a.second[dim] < b.first[dim] || b.second[dim] < a.first[dim])
                return false;
        }
        return true;
    }

    // Kernel between the two boxes' interpolation nodes, SVD-truncated into truncated_u (n_interp x rank)
    // and truncated_vt (rank x n_interp). Boxes are inflated and taken by value: theia's node functions want
    // non-const pointers.
    bool truncate_kernel(int M, int N, int L, int n_interp, box_type target_box, box_type source_box, underlying_type<CoefficientPrecision> epsilon, int reqrank, Matrix<CoefficientPrecision> &truncated_u, Matrix<CoefficientPrecision> &truncated_vt) const {
        if (boxes_intersect(target_box, source_box)) {
            return false;
        }

        // Compute interpolation matrix
        std::vector<std::array<CoordinatePrecision, dimension>> px(n_interp), py(n_interp);
        Matrix<CoefficientPrecision> K(n_interp, n_interp);
        theia::get_multivariate_interp_nodes<dimension, CoordinatePrecision, 0>(L, target_box.first.data(), target_box.second.data(), px.data());
        theia::get_multivariate_interp_nodes<dimension, CoordinatePrecision, 0>(L, source_box.first.data(), source_box.second.data(), py.data());
        theia::get_symbolic_matrix<dimension, CoordinatePrecision, CoefficientPrecision, kernel_type>(px.data(), py.data(), n_interp, n_interp, K.data(), &m_kernel);

        // truncated SVD
        std::vector<underlying_type<CoefficientPrecision>> singular_values(n_interp);
        Matrix<CoefficientPrecision> u(n_interp, n_interp);
        Matrix<CoefficientPrecision> vt(n_interp, n_interp);
        int truncated_rank = SVD_truncation(K, epsilon, u, vt, singular_values);
        if (reqrank > 0)
            truncated_rank = std::min({reqrank, n_interp, M, N});
        if (check_size && (std::size_t(truncated_rank) * std::size_t(M + N) > std::size_t(M) * std::size_t(N))) {
            return false;
        }
        truncated_u.resize(n_interp, truncated_rank);
        truncated_vt.resize(truncated_rank, n_interp);
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
        return true;
    }

    static void assemble_from_interpolation_matrices(const Matrix<CoefficientPrecision> &U_tilde, const Matrix<CoefficientPrecision> &V_tilde, const Matrix<CoefficientPrecision> &truncated_u, const Matrix<CoefficientPrecision> &truncated_vt, LowRankMatrix<CoefficientPrecision> &lrmat) {
        auto &U = lrmat.get_U();
        auto &V = lrmat.get_V();
        U.resize(U_tilde.nb_cols(), truncated_u.nb_cols());
        V.resize(truncated_vt.nb_rows(), V_tilde.nb_cols());
        add_matrix_matrix_product('T', 'N', CoefficientPrecision(1), U_tilde, truncated_u, CoefficientPrecision(0), U);
        add_matrix_matrix_product('N', 'N', CoefficientPrecision(1), truncated_vt, V_tilde, CoefficientPrecision(0), V);
    }

    bool compute_low_rank_approximation(int M, int N, int row_offset, int col_offset, underlying_type<CoefficientPrecision> epsilon, int reqrank, LowRankMatrix<CoefficientPrecision> &lrmat) const {
        auto target_node = m_target_nodes.find({row_offset, M});
        auto source_node = source_nodes().find({col_offset, N});
        bool tree_block  = target_node != m_target_nodes.end() && source_node != source_nodes().end();
        if (tree_block && epsilon < m_tree_epsilon) {
            // The trees' interpolation order was chosen for m_tree_epsilon and cannot meet a tighter one. A looser one
            // is fine: the SVD of the kernel truncates at the block's epsilon.
            std::ostringstream message;
            message << "HCA: block epsilon " << epsilon << " is tighter than the epsilon " << m_tree_epsilon << " the trees were precomputed with; computing this block directly.";
            Logger::get_instance().log(LogLevel::ERROR, message.str());
            tree_block = false;
        }
        if (tree_block) {
            Matrix<CoefficientPrecision> truncated_u, truncated_vt;
            if (!truncate_kernel(M, N, m_tree_L, m_tree_n_interp, target_node->second.box, source_node->second.box, epsilon, reqrank, truncated_u, truncated_vt)) {
                return false;
            }
            if (m_tree_mode == ReinterpolationMode::TensorLowRankFactors) {
                int rank                     = truncated_u.nb_cols();
                const node_data &target_data = target_node->second;
                const node_data &source_data = source_node->second;
                std::vector<CoefficientPrecision> particle_values(std::size_t(std::max(target_data.p2m.cols(), source_data.p2m.cols())) * rank);
                Matrix<CoefficientPrecision> &U = lrmat.get_U();
                U.resize(M, rank); // zero-filled: apply_w accumulates
                // gemTm is a friend of theia::TensorP2M, found by argument-dependent lookup
                gemTm(target_data.p2m, truncated_u.data(), particle_values.data(), rank);
                apply_w(m_target_nodes, target_node->first, particle_values.data(), target_data.p2m.cols(), rank, 0, U, row_offset);
                Matrix<CoefficientPrecision> truncated_v(m_tree_n_interp, rank), V_transposed(N, rank);
                transpose(truncated_vt, truncated_v);
                gemTm(source_data.p2m, truncated_v.data(), particle_values.data(), rank);
                apply_w(source_nodes(), source_node->first, particle_values.data(), source_data.p2m.cols(), rank, 0, V_transposed, col_offset);
                lrmat.get_V().resize(rank, N);
                transpose(V_transposed, lrmat.get_V());
                return true;
            }
            if (m_tree_mode == ReinterpolationMode::LowRankFactors) {
                int rank = truncated_u.nb_cols();
                lrmat.get_U().resize(M, rank);
                std::vector<CoefficientPrecision> workspace(std::size_t(std::max(target_node->second.height, source_node->second.height)) * m_tree_n_interp * rank);
                push_down(m_target_nodes, target_node->first, truncated_u.data(), rank, workspace.data(), lrmat.get_U(), row_offset);
                Matrix<CoefficientPrecision> truncated_v(m_tree_n_interp, rank), V_transposed(N, rank);
                transpose(truncated_vt, truncated_v);
                push_down(source_nodes(), source_node->first, truncated_v.data(), rank, workspace.data(), V_transposed, col_offset);
                lrmat.get_V().resize(rank, N);
                transpose(V_transposed, lrmat.get_V());
                return true;
            }
            Matrix<CoefficientPrecision> U_tilde(m_tree_n_interp, M), V_tilde(m_tree_n_interp, N);
            reinterpolate(m_target_nodes, target_node->first, U_tilde.data());
            reinterpolate(source_nodes(), source_node->first, V_tilde.data());
            assemble_from_interpolation_matrices(U_tilde, V_tilde, truncated_u, truncated_vt, lrmat);
            return true;
        }

        // Direct path: the block is not a node of the precomputed trees (or there are none).
        int L               = interpolation_order(epsilon);
        int n_interp        = number_of_interpolation_nodes<dimension>(L);
        box_type target_box = m_target->box(row_offset, M);
        box_type source_box = m_source->box(col_offset, N);
        Matrix<CoefficientPrecision> truncated_u, truncated_vt;
        if (!truncate_kernel(M, N, L, n_interp, target_box, source_box, epsilon, reqrank, truncated_u, truncated_vt)) {
            return false;
        }
        assemble_from_interpolation_matrices(m_target->interpolation_matrix(L, target_box, row_offset, M), m_source->interpolation_matrix(L, source_box, col_offset, N), truncated_u, truncated_vt, lrmat);
        return true;
    }

    void build_node_table(const discretization_type &discretization, std::map<node_key, node_data> &nodes) {
        postorder_tree_traversal(discretization.cluster(), [&](const Cluster<CoordinatePrecision> &node) {
            const auto &children = node.get_children();
            if (children.size() == 1) {
                return; // same {offset, size} and box as its only child, whose record stands
            }
            node_data data;
            if (children.empty()) {
                data.box = discretization.box(node.get_offset(), node.get_size());
                if (m_tree_mode == ReinterpolationMode::TensorLowRankFactors) {
                    discretization.tensor_interpolation(m_tree_L, data.box, node.get_offset(), node.get_size(), data.p2m, data.w);
                } else {
                    data.leaf_matrix = discretization.interpolation_matrix(m_tree_L, data.box, node.get_offset(), node.get_size());
                }
            } else {
                data.box.first.fill(std::numeric_limits<CoordinatePrecision>::max());
                data.box.second.fill(std::numeric_limits<CoordinatePrecision>::lowest());
                for (const auto &child : children) {
                    const box_type &child_box = nodes.at({child->get_offset(), child->get_size()}).box;
                    for (int dim = 0; dim < dimension; dim++) {
                        data.box.first[dim]  = std::min(data.box.first[dim], child_box.first[dim]);
                        data.box.second[dim] = std::max(data.box.second[dim], child_box.second[dim]);
                    }
                }
                std::array<int, dimension> Ls;
                Ls.fill(m_tree_L);
                for (const auto &child : children) {
                    node_key child_key{child->get_offset(), child->get_size()};
                    box_type child_box  = nodes.at(child_key).box;
                    box_type parent_box = data.box;
                    theia::Kron<dimension, CoefficientPrecision> A;
                    theia::M2M<dimension, CoordinatePrecision, CoefficientPrecision, 0>(Ls.data(), parent_box.first.data(), parent_box.second.data(), Ls.data(), child_box.first.data(), child_box.second.data(), A, theia::op_interpolation<dimension, CoordinatePrecision, CoefficientPrecision, 0>{});
                    A.prcmp(); // builds the gemm and gemTm permutations now so concurrent applies never write
                    data.children.emplace_back(child_key, std::move(A));
                    data.height = std::max(data.height, nodes.at(child_key).height + 1);
                }
                if (m_tree_mode == ReinterpolationMode::TensorLowRankFactors) {
                    // Mixed product: Kron * TensorP2M is the TensorP2M with factors K_d * A_d, i.e. the child's
                    // particles seen on this box. This node's particles are its children's, in child order.
                    std::vector<theia::TensorP2M<dimension, CoefficientPrecision>> lifted(data.children.size());
                    int number_of_particles = 0;
                    for (std::size_t c = 0; c < data.children.size(); c++) {
                        theia::gemm(data.children[c].second, nodes.at(data.children[c].first).p2m, lifted[c]);
                        number_of_particles += lifted[c].cols();
                    }
                    data.p2m.resize(Ls.data(), number_of_particles);
                    int first_particle = 0;
                    for (auto &child_p2m : lifted) {
                        for (int d = 0; d < dimension; d++) {
                            std::copy_n(child_p2m.factor(d), std::size_t(m_tree_L) * child_p2m.cols(), data.p2m.factor(d) + std::size_t(m_tree_L) * first_particle);
                        }
                        first_particle += child_p2m.cols();
                    }
                }
            }
            nodes[{node.get_offset(), node.get_size()}] = std::move(data);
        });
    }

    // Writes the node's n_interp x size dof-to-interpolation matrix into out (column major).
    void reinterpolate(std::map<node_key, node_data> &nodes, const node_key &key, CoefficientPrecision *out) const {
        node_data &data = nodes.at(key);
        if (data.children.empty()) {
            std::copy_n(data.leaf_matrix.data(), std::size_t(m_tree_n_interp) * key.second, out);
            return;
        }
        for (auto &child : data.children) {
            const node_key &child_key   = child.first;
            CoefficientPrecision *slice = out + std::size_t(child_key.first - key.first) * m_tree_n_interp;
            reinterpolate(nodes, child_key, slice);
            // In place: Kron::apply copies its whole input before writing the output.
            // gemm is a friend of theia::Kron, found by argument-dependent lookup.
            gemm(child.second, slice, slice, child_key.second);
        }
    }

    // Applies the transposed M2M chain to coefficients (n_interp x rank, on this node's grid) down to the leaves,
    // and writes leaf_matrix^T * coefficients into this node's rows of out (block_size x rank). workspace holds
    // height slices of n_interp x rank, one per level below this node.
    void push_down(std::map<node_key, node_data> &nodes, const node_key &key, CoefficientPrecision *coefficients, int rank, CoefficientPrecision *workspace, Matrix<CoefficientPrecision> &out, int block_offset) const {
        node_data &data = nodes.at(key);
        if (data.children.empty()) {
            char transa = 'T', transb = 'N';
            int leaf_size = key.second, n_interp = m_tree_n_interp, ld_out = out.nb_rows();
            CoefficientPrecision one(1), zero(0);
            Blas<CoefficientPrecision>::gemm(&transa, &transb, &leaf_size, &rank, &n_interp, &one, data.leaf_matrix.data(), &n_interp, coefficients, &n_interp, &zero, out.data() + (key.first - block_offset), &ld_out);
            return;
        }
        for (auto &child : data.children) {
            // gemTm is a friend of theia::Kron, found by argument-dependent lookup
            gemTm(child.second, coefficients, workspace, rank);
            push_down(nodes, child.first, workspace, rank, workspace + std::size_t(m_tree_n_interp) * rank, out, block_offset);
        }
    }

    // Adds W^T * particle_values (block particles x rank, leading dimension ld) into this node's rows of out.
    // particle_offset is this node's first particle in the block's particle ordering.
    void apply_w(const std::map<node_key, node_data> &nodes, const node_key &key, const CoefficientPrecision *particle_values, int ld, int rank, int particle_offset, Matrix<CoefficientPrecision> &out, int block_offset) const {
        const node_data &data = nodes.at(key);
        if (data.children.empty()) {
            int first_row = key.first - block_offset;
            for (const auto &entry : data.w) {
                for (int c = 0; c < rank; c++) {
                    out(first_row + entry.dof, c) += entry.value * particle_values[std::size_t(c) * ld + particle_offset + entry.particle];
                }
            }
            return;
        }
        for (const auto &child : data.children) {
            apply_w(nodes, child.first, particle_values, ld, rank, particle_offset, out, block_offset);
            particle_offset += nodes.at(child.first).p2m.cols();
        }
    }

  public:
    // Offsets are read through each discretization's cluster permutation: it must be the tree the blocks come from.
    HCA(kernel_type kernel, std::shared_ptr<const discretization_type> target, std::shared_ptr<const discretization_type> source) : m_kernel(kernel), m_target(std::move(target)), m_source(std::move(source)) {}

    // Also precomputes the leaves of both discretizations' cluster trees for epsilon, once if target == source.
    HCA(kernel_type kernel, std::shared_ptr<const discretization_type> target, std::shared_ptr<const discretization_type> source, underlying_type<CoefficientPrecision> epsilon, ReinterpolationMode mode = ReinterpolationMode::InterpolationMatrices) : m_kernel(kernel), m_target(std::move(target)), m_source(std::move(source)) {
        m_tree_mode     = mode;
        m_tree_epsilon  = epsilon;
        m_tree_L        = interpolation_order(epsilon);
        m_tree_n_interp = number_of_interpolation_nodes<dimension>(m_tree_L);

        m_symmetric = m_target == m_source;
        build_node_table(*m_target, m_target_nodes);
        if (!m_symmetric) {
            build_node_table(*m_source, m_source_nodes);
        }
    }

    bool copy_low_rank_approximation(int M, int N, int row_offset, int col_offset, LowRankMatrix<CoefficientPrecision> &lrmat) const override {
        return compute_low_rank_approximation(M, N, row_offset, col_offset, lrmat.get_epsilon(), -1, lrmat);
    }

    bool copy_low_rank_approximation(int M, int N, int row_offset, int col_offset, int reqrank, LowRankMatrix<CoefficientPrecision> &lrmat) const override {
        return compute_low_rank_approximation(M, N, row_offset, col_offset, lrmat.get_epsilon(), reqrank, lrmat);
    }
};

} // namespace htool

#endif
