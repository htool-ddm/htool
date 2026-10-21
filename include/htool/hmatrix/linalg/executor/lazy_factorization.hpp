#ifndef HTOOL_HMATRIX_LINALG_EXECUTOR_LAZY_FACTORIZATION_HPP
#define HTOOL_HMATRIX_LINALG_EXECUTOR_LAZY_FACTORIZATION_HPP

#include "../../../basic_types/tree.hpp"               // for preorder_tree_traversal
#include "../../../clustering/cluster_node.hpp"        // for Cluster
#include "../../../matrix/linalg/factorization.hpp"    // for lu_factorization
#include "../../../matrix/linalg/scale.hpp"            // for scale
#include "../../../matrix/matrix.hpp"                  // for Matrix
#include "../../../misc/logger.hpp"                    // for Logger
#include "../../../wrappers/wrapper_blas.hpp"          // for Blas
#include "../../hmatrix.hpp"                           // for HMatrix, copy_to_dense
#include "../../interfaces/virtual_generator.hpp"      // for VirtualInternalGenerator
#include "../../lrmat/lrmat.hpp"                       // for LowRankMatrix
#include "../../lrmat/partialACA.hpp"                  // for partialACA
#include "../../lrmat/utils/SVD_recompression.hpp"     // for SVD_recompression
#include "../omp_task/add_hmatrix_hmatrix_product.hpp" // for fill_clusters
#include "../seq/add_hmatrix_hmatrix_product.hpp"      // for internal_add_hmatrix_hmatrix_product
#include "../seq/add_hmatrix_matrix_product.hpp"       // for internal_add_hmatrix_matrix_product
#include "../seq/add_hmatrix_vector_product.hpp"       // for internal_add_hmatrix_vector_product
#include "../seq/add_matrix_hmatrix_product.hpp"       // for internal_add_matrix_hmatrix_product
#include "../seq/triangular_hmatrix_hmatrix_solve.hpp" // for internal_triangular_hmatrix_hmatrix_solve
#include "thread_executor.hpp"                         // for ExecutorAccess
#include <algorithm>                                   // for copy_n, fill
#include <deque>                                       // for deque
#include <memory>                                      // for shared_ptr
#include <unordered_map>                               // for unordered_map
#include <utility>                                     // for pair
#include <vector>                                      // for vector

namespace htool {

/**
 * @brief An exact low-rank term \f$UV\f$ of size the block it was computed for: the product \f$-xy\f$ of two blocks of the factors, one of
 * them being a leaf. One of \f$U\f$ and \f$V\f$ is usually a factor of this leaf (not a copy), the other one is computed (own_U or own_V).
 *
 * It is computed by a task, and read by the flushes of the leaves below this block (see LazyLUDriver), through restrictions (CarryTerm).
 */
template <typename CoefficientPrecision>
struct LazyLowRankTerm {
    const Matrix<CoefficientPrecision> *U = nullptr;
    const Matrix<CoefficientPrecision> *V = nullptr;
    Matrix<CoefficientPrecision> own_U, own_V;

    LazyLowRankTerm() = default;
    // U or V may point to own_U or own_V
    LazyLowRankTerm(const LazyLowRankTerm &)            = delete;
    LazyLowRankTerm &operator=(const LazyLowRankTerm &) = delete;

    int rank() const { return U->nb_cols(); }
};

/**
 * @brief Restriction of a LazyLowRankTerm to a block below the one it was computed for: its rows starting at row_offset, and its columns
 * starting at col_offset, the size of the restriction being the one of this block.
 */
template <typename CoefficientPrecision>
struct CarryTerm {
    std::shared_ptr<LazyLowRankTerm<CoefficientPrecision>> term;
    int row_offset;
    int col_offset;
};

/**
 * @brief The operator \f$UV - \sum_p x_p y_p\f$ of size M x N, known through its rows and columns, compressed by partialACA when an
 * admissible leaf of the factors is flushed (see LazyLUDriver).
 *
 * A row is computed as \f$U(i,:)V - \sum_p x_p(i,:) y_p\f$: the row of \f$x_p\f$ is extracted from the leaves of \f$x_p\f$ it meets, and
 * multiplied by \f$y_p\f$ (one product with the leaves of \f$y_p\f$). A column is computed symmetrically.
 */
template <typename CoefficientPrecision, typename CoordinatePrecision>
class LazyLeafOperator : public VirtualInternalGenerator<CoefficientPrecision> {
    using HMatrixType = HMatrix<CoefficientPrecision, CoordinatePrecision>;

    struct Product {
        const HMatrixType *x, *y;
        std::vector<const HMatrixType *> x_leaves, y_leaves;
    };

    int m_nb_rows, m_nb_cols;
    const Matrix<CoefficientPrecision> &m_U; // M x R
    const Matrix<CoefficientPrecision> &m_V; // R x N
    std::vector<Product> m_products;
    mutable std::vector<CoefficientPrecision> m_middle;

    static std::vector<const HMatrixType *> leaves_of(const HMatrixType &hmatrix) {
        std::vector<const HMatrixType *> leaves;
        preorder_tree_traversal(hmatrix, [&leaves](const HMatrixType &node) {
            if (!node.is_hierarchical()) {
                leaves.push_back(&node);
            }
        });
        return leaves;
    }

    // out <- row (trans='N') or column (trans='T') index of hmatrix, index being local
    static void copy_row_or_column(char trans, const HMatrixType &hmatrix, int index, CoefficientPrecision *out) {
        if (hmatrix.is_hierarchical()) {
            int global_index = index + (trans == 'N' ? hmatrix.get_target_cluster().get_offset() : hmatrix.get_source_cluster().get_offset());
            for (const auto &child : hmatrix.get_children()) {
                const Cluster<CoordinatePrecision> &cluster = trans == 'N' ? child->get_target_cluster() : child->get_source_cluster();
                if (cluster.get_offset() <= global_index && global_index < cluster.get_offset() + cluster.get_size()) {
                    int out_offset = trans == 'N' ? child->get_source_cluster().get_offset() - hmatrix.get_source_cluster().get_offset() : child->get_target_cluster().get_offset() - hmatrix.get_target_cluster().get_offset();
                    copy_row_or_column(trans, *child, global_index - cluster.get_offset(), out + out_offset);
                }
            }
        } else if (hmatrix.is_dense()) {
            const Matrix<CoefficientPrecision> &dense = *hmatrix.get_dense_data();
            if (trans == 'N') {
                for (int j = 0; j < dense.nb_cols(); j++) {
                    out[j] = dense(index, j);
                }
            } else {
                std::copy_n(dense.data() + index * dense.nb_rows(), dense.nb_rows(), out);
            }
        } else {
            const LowRankMatrix<CoefficientPrecision> &lrmat = *hmatrix.get_low_rank_data();
            const Matrix<CoefficientPrecision> &U            = lrmat.get_U();
            const Matrix<CoefficientPrecision> &V            = lrmat.get_V();
            int rank = lrmat.rank_of(), M = U.nb_rows(), N = V.nb_cols(), inc = 1;
            CoefficientPrecision one(1), zero(0);
            if (rank == 0) {
                std::fill(out, out + (trans == 'N' ? N : M), zero);
            } else if (trans == 'N') { // out = V^T U(index,:)
                char t = 'T';
                Blas<CoefficientPrecision>::gemv(&t, &rank, &N, &one, V.data(), &rank, U.data() + index, &M, &zero, out, &inc);
            } else { // out = U V(:,index)
                char t = 'N';
                Blas<CoefficientPrecision>::gemv(&t, &M, &rank, &one, U.data(), &M, V.data() + index * rank, &inc, &zero, out, &inc);
            }
        }
    }

    // out <- out + alpha op(hmatrix) in, through its leaves
    static void add_product(char trans, CoefficientPrecision alpha, const HMatrixType &hmatrix, const std::vector<const HMatrixType *> &leaves, const CoefficientPrecision *in, CoefficientPrecision *out) {
        int target_offset = hmatrix.get_target_cluster().get_offset();
        int source_offset = hmatrix.get_source_cluster().get_offset();
        for (const HMatrixType *leaf : leaves) {
            int leaf_target_offset = leaf->get_target_cluster().get_offset() - target_offset;
            int leaf_source_offset = leaf->get_source_cluster().get_offset() - source_offset;
            if (trans == 'N') {
                internal_add_hmatrix_vector_product('N', alpha, *leaf, in + leaf_source_offset, CoefficientPrecision(1), out + leaf_target_offset);
            } else {
                internal_add_hmatrix_vector_product('T', alpha, *leaf, in + leaf_target_offset, CoefficientPrecision(1), out + leaf_source_offset);
            }
        }
    }

  public:
    LazyLeafOperator(int nb_rows, int nb_cols, const Matrix<CoefficientPrecision> &U, const Matrix<CoefficientPrecision> &V, const std::vector<std::pair<const HMatrixType *, const HMatrixType *>> &products) : m_nb_rows(nb_rows), m_nb_cols(nb_cols), m_U(U), m_V(V) {
        for (const auto &product : products) {
            m_products.push_back({product.first, product.second, leaves_of(*product.first), leaves_of(*product.second)});
        }
    }

    void copy_submatrix(int M, int N, int row_offset, int col_offset, CoefficientPrecision *ptr) const override {
        int rank = m_U.nb_cols(), inc = 1;
        CoefficientPrecision one(1), zero(0), minus_one(-1);
        if (M == 1) {
            if (rank > 0) {
                char t = 'T';
                Blas<CoefficientPrecision>::gemv(&t, &rank, &N, &one, m_V.data() + col_offset * rank, &rank, m_U.data() + row_offset, &m_nb_rows, &zero, ptr, &inc);
            } else {
                std::fill(ptr, ptr + N, zero);
            }
            for (const Product &product : m_products) {
                m_middle.resize(product.x->nb_cols());
                copy_row_or_column('N', *product.x, row_offset, m_middle.data());
                if (col_offset == 0 && N == m_nb_cols) {
                    add_product('T', minus_one, *product.y, product.y_leaves, m_middle.data(), ptr);
                } else {
                    std::vector<CoefficientPrecision> row(m_nb_cols, zero);
                    add_product('T', minus_one, *product.y, product.y_leaves, m_middle.data(), row.data());
                    for (int j = 0; j < N; j++) {
                        ptr[j] += row[col_offset + j];
                    }
                }
            }
        } else {
            // Column by column, each one as a column of the operator restricted to the rows asked for
            std::vector<CoefficientPrecision> column(m_nb_rows);
            for (int j = 0; j < N; j++) {
                if (rank > 0) {
                    char t = 'N';
                    Blas<CoefficientPrecision>::gemv(&t, &m_nb_rows, &rank, &one, m_U.data(), &m_nb_rows, m_V.data() + (col_offset + j) * rank, &inc, &zero, column.data(), &inc);
                } else {
                    std::fill(column.begin(), column.end(), zero);
                }
                for (const Product &product : m_products) {
                    m_middle.resize(product.y->nb_rows());
                    copy_row_or_column('T', *product.y, col_offset + j, m_middle.data());
                    add_product('N', minus_one, *product.x, product.x_leaves, m_middle.data(), column.data());
                }
                std::copy_n(column.data() + row_offset, M, ptr + j * M);
            }
        }
    }
};

/**
 * @brief The lazy hierarchical LU factorization of HMatrices.jl, where every update is deferred until the block it targets is factorized
 * or solved for, with its numeric work run by an executor.
 *
 * By the block Crout identity, every block of the factors is determined by a single Schur complement, a sum of products of blocks of the
 * factors that precede it. Instead of evaluating these products one at a time and recompressing after each one (sequential_lu_factorization),
 * each node carries the products it still owes (pending), and the exact low-rank terms already evaluated for it (carry). A node is flushed
 * right before it is factorized or solved for:
 *   - internal node: a pending product with a low-rank leaf factor has an exact low-rank form, which is computed once, here, and shared by
 *     the leaves below (carry); the other products are split into the products of the children; the carry is restricted to the children;
 *   - dense leaf: the carry and the products are added exactly;
 *   - admissible (low-rank) leaf: its data, its carry and the products with an exact low-rank form are written side by side as \f$UV\f$, and
 *     \f$UV - \sum_p x_p y_p\f$ (the other products) is compressed once by partialACA (see LazyLeafOperator).
 * Every admissible leaf of the factors is thus compressed once.
 *
 * The driver does the structural work (pushing and splitting products, restricting low-rank terms) itself, and hands the numeric work to
 * the executor: the low-rank terms of a node, the flush of a leaf, the dense LU of a diagonal leaf and the triangular solves on a leaf. A leaf
 * is written by at most one flush followed by one factorization or solve, a low-rank term by one task, and a block is only read once final,
 * which is the access pattern ThreadExecutor relies on.
 *
 * The factors are stored as by sequential_lu_factorization, so that lu_solve can be used with them. The storage type of every leaf is kept.
 * If recompress is true, the result of each compression is recompressed by SVD_recompression (ACA overestimates the rank).
 */
template <typename Executor, typename CoefficientPrecision, typename CoordinatePrecision>
class LazyLUDriver {
    using HMatrixType = HMatrix<CoefficientPrecision, CoordinatePrecision>;
    using ClusterType = Cluster<CoordinatePrecision>;
    using Access      = ExecutorAccess<CoefficientPrecision, CoordinatePrecision>;
    using Term        = LazyLowRankTerm<CoefficientPrecision>;
    using Carry       = CarryTerm<CoefficientPrecision>;
    using Products    = std::vector<std::pair<const HMatrixType *, const HMatrixType *>>;

    struct PendingUpdates {
        Products products;
        std::vector<Carry> carry;
    };

    Executor &m_executor;
    bool m_recompress;
    std::unordered_map<const HMatrixType *, PendingUpdates> m_pending;

    static std::vector<const ClusterType *> children_of(const ClusterType &cluster) {
        std::vector<const ClusterType *> clusters;
        fill_clusters(cluster, false, clusters);
        return clusters;
    }

    void push(const HMatrixType &C, const HMatrixType &x, const HMatrixType &y) {
        m_pending[&C].products.emplace_back(&x, &y);
    }

    // Whether the product x*y targeting the internal node C can be split into the products of the children of x and y targeting the children of C.
    static bool is_splittable(const HMatrixType &C, const HMatrixType &x, const HMatrixType &y) {
        std::vector<const ClusterType *> middle_clusters = children_of(x.get_source_cluster());
        for (const auto &C_child : C.get_children()) {
            for (const ClusterType *middle_cluster : middle_clusters) {
                if (x.get_child_or_this(C_child->get_target_cluster(), *middle_cluster) == nullptr || y.get_child_or_this(*middle_cluster, C_child->get_source_cluster()) == nullptr) {
                    return false;
                }
            }
        }
        return true;
    }

    // term <- -x*y, as an exact low-rank term. x or y is a low-rank leaf, or, if the product cannot be split (internal node) or both are dense
    // (leaf), a dense leaf.
    static void compute_low_rank_term(const HMatrixType &x, const HMatrixType &y, Term &term) {
        CoefficientPrecision minus_one(-1), zero(0);
        if (x.is_low_rank()) { // -x*y = U_x (-V_x y)
            const LowRankMatrix<CoefficientPrecision> &lrmat = *x.get_low_rank_data();
            term.U                                           = &lrmat.get_U();
            term.own_V.resize(lrmat.rank_of(), y.nb_cols());
            internal_add_matrix_hmatrix_product('N', 'N', minus_one, lrmat.get_V(), y, zero, term.own_V);
            term.V = &term.own_V;
        } else if (y.is_low_rank()) { // -x*y = (-x U_y) V_y
            const LowRankMatrix<CoefficientPrecision> &lrmat = *y.get_low_rank_data();
            term.own_U.resize(x.nb_rows(), lrmat.rank_of());
            internal_add_hmatrix_matrix_product('N', 'N', minus_one, x, lrmat.get_U(), zero, term.own_U);
            term.U = &term.own_U;
            term.V = &lrmat.get_V();
        } else if (x.is_dense()) { // -x*y = x (-y)
            term.U = x.get_dense_data();
            term.own_V.resize(y.nb_rows(), y.nb_cols());
            copy_to_dense(y, term.own_V.data());
            scale(minus_one, term.own_V);
            term.V = &term.own_V;
        } else { // -x*y = (-x) y
            term.own_U.resize(x.nb_rows(), x.nb_cols());
            copy_to_dense(x, term.own_U.data());
            scale(minus_one, term.own_U);
            term.U = &term.own_U;
            if (y.is_dense()) {
                term.V = y.get_dense_data();
            } else {
                term.own_V.resize(y.nb_rows(), y.nb_cols());
                copy_to_dense(y, term.own_V.data());
                term.V = &term.own_V;
            }
        }
    }

    static bool has_low_rank_form_at_leaf(const HMatrixType &x, const HMatrixType &y) {
        return x.is_low_rank() || y.is_low_rank() || (x.is_dense() && y.is_dense());
    }

    static std::vector<Access> accesses_of(const Products &products, const std::vector<Carry> &carry) {
        std::vector<Access> accesses;
        for (const auto &product : products) {
            accesses.emplace_back(*product.first);
            accesses.emplace_back(*product.second);
        }
        for (const Carry &carry_term : carry) {
            accesses.emplace_back(static_cast<const void *>(carry_term.term.get()));
        }
        return accesses;
    }

    // Applies the updates pending at C (not those pending at its descendants).
    void flush(HMatrixType &C) {
        auto it = m_pending.find(&C);
        if (it == m_pending.end()) {
            return;
        }
        auto updates = std::make_shared<PendingUpdates>(std::move(it->second));
        m_pending.erase(it);

        if (C.is_hierarchical()) {
            // Products with an exact low-rank form, computed by one task, and shared by the leaves below C
            Products splittable_products, low_rank_products;
            for (const auto &product : updates->products) {
                if (!product.first->is_low_rank() && !product.second->is_low_rank() && is_splittable(C, *product.first, *product.second)) {
                    splittable_products.push_back(product);
                } else {
                    low_rank_products.push_back(product);
                }
            }
            if (!low_rank_products.empty()) {
                auto terms = std::make_shared<std::vector<std::shared_ptr<Term>>>();
                std::vector<Access> writes;
                for (std::size_t p = 0; p < low_rank_products.size(); p++) {
                    terms->push_back(std::make_shared<Term>());
                    updates->carry.push_back({terms->back(), 0, 0});
                    writes.emplace_back(static_cast<const void *>(terms->back().get()));
                }
                auto products             = std::make_shared<Products>(std::move(low_rank_products));
                std::vector<Access> reads = accesses_of(*products, {});
                m_executor.spawn(
                    [products, terms]() {
                        for (std::size_t p = 0; p < products->size(); p++) {
                            compute_low_rank_term(*(*products)[p].first, *(*products)[p].second, *(*terms)[p]);
                        }
                    },
                    reads,
                    writes);
            }
            // The other products and the carry are handed to the children
            for (const auto &product : splittable_products) {
                const HMatrixType &x                             = *product.first;
                const HMatrixType &y                             = *product.second;
                std::vector<const ClusterType *> middle_clusters = children_of(x.get_source_cluster());
                for (const auto &C_child : C.get_children()) {
                    for (const ClusterType *middle_cluster : middle_clusters) {
                        push(*C_child, *x.get_child_or_this(C_child->get_target_cluster(), *middle_cluster), *y.get_child_or_this(*middle_cluster, C_child->get_source_cluster()));
                    }
                }
            }
            if (!updates->carry.empty()) {
                for (const auto &C_child : C.get_children()) {
                    int row_shift             = C_child->get_target_cluster().get_offset() - C.get_target_cluster().get_offset();
                    int col_shift             = C_child->get_source_cluster().get_offset() - C.get_source_cluster().get_offset();
                    std::vector<Carry> &carry = m_pending[C_child.get()].carry;
                    for (const Carry &carry_term : updates->carry) {
                        carry.push_back({carry_term.term, carry_term.row_offset + row_shift, carry_term.col_offset + col_shift});
                    }
                }
            }
        } else {
            HMatrixType *leaf         = &C;
            std::vector<Access> reads = accesses_of(updates->products, updates->carry);
            if (C.is_dense()) {
                m_executor.spawn([leaf, updates]() { flush_dense_leaf(*leaf, *updates); }, reads, {Access(C)});
            } else {
                bool recompress = m_recompress;
                m_executor.spawn([leaf, updates, recompress]() { flush_low_rank_leaf(*leaf, *updates, recompress); }, reads, {Access(C)});
            }
        }
    }

    static void flush_dense_leaf(HMatrixType &C, const PendingUpdates &updates) {
        Matrix<CoefficientPrecision> &dense = *C.get_dense_data();
        int M = dense.nb_rows(), N = dense.nb_cols();
        CoefficientPrecision one(1);
        char t = 'N';
        for (const Carry &carry_term : updates.carry) {
            const Term &term = *carry_term.term;
            int rank         = term.rank();
            if (rank > 0) {
                int ldu = term.U->nb_rows();
                Blas<CoefficientPrecision>::gemm(&t, &t, &M, &N, &rank, &one, term.U->data() + carry_term.row_offset, &ldu, term.V->data() + carry_term.col_offset * rank, &rank, &one, dense.data(), &M);
            }
        }
        for (const auto &product : updates.products) {
            internal_add_hmatrix_hmatrix_product('N', 'N', CoefficientPrecision(-1), *product.first, *product.second, one, dense);
        }
    }

    static void flush_low_rank_leaf(HMatrixType &C, const PendingUpdates &updates, bool recompress) {
        LowRankMatrix<CoefficientPrecision> &lrmat = *C.get_low_rank_data();
        int M = C.nb_rows(), N = C.nb_cols();

        // The products with an exact low-rank form are evaluated in this form, the others are only sampled by the compression
        Products implicit_products;
        std::deque<Term> exact_terms;
        for (const auto &product : updates.products) {
            if (has_low_rank_form_at_leaf(*product.first, *product.second)) {
                exact_terms.emplace_back();
                compute_low_rank_term(*product.first, *product.second, exact_terms.back());
            } else {
                implicit_products.push_back(product);
            }
        }

        // All the low-rank terms, written side by side as U V
        int rank = lrmat.rank_of();
        for (const Carry &carry_term : updates.carry) {
            rank += carry_term.term->rank();
        }
        for (const Term &term : exact_terms) {
            rank += term.rank();
        }
        Matrix<CoefficientPrecision> U(M, rank), V(rank, N);
        int k       = 0;
        auto append = [&](const Matrix<CoefficientPrecision> &U_term, int row_offset, const Matrix<CoefficientPrecision> &V_term, int col_offset) {
            int r = U_term.nb_cols();
            for (int l = 0; l < r; l++) {
                std::copy_n(U_term.data() + l * U_term.nb_rows() + row_offset, M, U.data() + (k + l) * M);
            }
            for (int j = 0; j < N; j++) {
                std::copy_n(V_term.data() + (col_offset + j) * r, r, V.data() + j * rank + k);
            }
            k += r;
        };
        append(lrmat.get_U(), 0, lrmat.get_V(), 0);
        for (const Carry &carry_term : updates.carry) {
            append(*carry_term.term->U, carry_term.row_offset, *carry_term.term->V, carry_term.col_offset);
        }
        for (const Term &term : exact_terms) {
            append(*term.U, 0, *term.V, 0);
        }

        // One compression of U V - sum of the implicit products
        LazyLeafOperator<CoefficientPrecision, CoordinatePrecision> leaf_operator(M, N, U, V, implicit_products);
        LowRankMatrix<CoefficientPrecision> compressed(M, N, lrmat.get_epsilon());
        bool is_compressed = partialACA<CoefficientPrecision>(leaf_operator).copy_low_rank_approximation(M, N, 0, 0, compressed);
        if (!is_compressed) {
            // Low rank not advantageous (or ACA failed): truncated SVD of the dense operator, keeping the low-rank storage of the leaf
            compressed.get_U().resize(M, N);
            leaf_operator.copy_submatrix(M, N, 0, 0, compressed.get_U().data());
            compressed.get_V().resize(N, N);
            for (int j = 0; j < N; j++) {
                compressed.get_V()(j, j) = CoefficientPrecision(1);
            }
            SVD_recompression(compressed);
        } else if (recompress) {
            SVD_recompression(compressed);
        }
        lrmat.get_U() = std::move(compressed.get_U());
        lrmat.get_V() = std::move(compressed.get_V());
    }

  public:
    explicit LazyLUDriver(Executor &executor, bool recompress = false) : m_executor(executor), m_recompress(recompress) {}

    void lu(HMatrixType &M) {
        flush(M);
        if (M.is_hierarchical()) {
            std::vector<const ClusterType *> clusters = children_of(M.get_target_cluster());
            for (std::size_t i = 0; i < clusters.size(); i++) {
                HMatrixType &M_ii = *M.get_child_or_this(*clusters[i], *clusters[i]);
                lu(M_ii);
                for (std::size_t j = i + 1; j < clusters.size(); j++) {
                    solve_lower(M_ii, *M.get_child_or_this(*clusters[i], *clusters[j]));
                    solve_upper(*M.get_child_or_this(*clusters[j], *clusters[i]), M_ii);
                }
                for (std::size_t j = i + 1; j < clusters.size(); j++) {
                    for (std::size_t k = i + 1; k < clusters.size(); k++) {
                        push(*M.get_child_or_this(*clusters[j], *clusters[k]), *M.get_child_or_this(*clusters[j], *clusters[i]), *M.get_child_or_this(*clusters[i], *clusters[k]));
                    }
                }
            }
        } else if (M.is_dense()) {
            HMatrixType *leaf = &M;
            m_executor.spawn([leaf]() { lu_factorization(*leaf->get_dense_data()); }, {}, {Access(M)});
        } else {
            Logger::get_instance().log(LogLevel::ERROR, "Operation is not implemented for lazy_lu_factorization (hmatrix is low-rank)"); // LCOV_EXCL_LINE
        }
    }

    // X <- L^-1 X, L being the unit lower triangle of the factorized diagonal block D
    void solve_lower(const HMatrixType &D, HMatrixType &X) {
        flush(X);
        if (!X.is_hierarchical()) {
            const HMatrixType *D_ptr = &D;
            HMatrixType *X_ptr       = &X;
            m_executor.spawn([D_ptr, X_ptr]() { internal_triangular_hmatrix_hmatrix_solve('L', 'L', 'N', 'U', CoefficientPrecision(1), *D_ptr, *X_ptr); }, {Access(D)}, {Access(X)});
        } else if (!D.is_hierarchical()) {
            // D is a dense diagonal block, whose cluster is kept by the children of X in a consistent block tree
            for (auto &child : X.get_children_with_ownership()) {
                solve_lower(D, *child);
            }
        } else {
            std::vector<const ClusterType *> rows    = children_of(D.get_target_cluster());
            std::vector<const ClusterType *> columns = children_of(X.get_source_cluster());
            for (const ClusterType *column : columns) {
                for (std::size_t i = 0; i < rows.size(); i++) {
                    HMatrixType &X_i = *X.get_child_or_this(*rows[i], *column);
                    solve_lower(*D.get_child_or_this(*rows[i], *rows[i]), X_i);
                    for (std::size_t j = i + 1; j < rows.size(); j++) {
                        push(*X.get_child_or_this(*rows[j], *column), *D.get_child_or_this(*rows[j], *rows[i]), X_i);
                    }
                }
            }
        }
    }

    // X <- X U^-1, U being the upper triangle of the factorized diagonal block D
    void solve_upper(HMatrixType &X, const HMatrixType &D) {
        flush(X);
        if (!X.is_hierarchical()) {
            const HMatrixType *D_ptr = &D;
            HMatrixType *X_ptr       = &X;
            m_executor.spawn([D_ptr, X_ptr]() { internal_triangular_hmatrix_hmatrix_solve('R', 'U', 'N', 'N', CoefficientPrecision(1), *D_ptr, *X_ptr); }, {Access(D)}, {Access(X)});
        } else if (!D.is_hierarchical()) {
            for (auto &child : X.get_children_with_ownership()) {
                solve_upper(*child, D);
            }
        } else {
            std::vector<const ClusterType *> columns = children_of(D.get_source_cluster());
            std::vector<const ClusterType *> rows    = children_of(X.get_target_cluster());
            for (const ClusterType *row : rows) {
                for (std::size_t i = 0; i < columns.size(); i++) {
                    HMatrixType &X_i = *X.get_child_or_this(*row, *columns[i]);
                    solve_upper(X_i, *D.get_child_or_this(*columns[i], *columns[i]));
                    for (std::size_t j = i + 1; j < columns.size(); j++) {
                        push(*X.get_child_or_this(*row, *columns[j]), X_i, *D.get_child_or_this(*columns[i], *columns[j]));
                    }
                }
            }
        }
    }
};

/**
 * @brief Lazy LU factorization of hmatrix (see LazyLUDriver), with its numeric work run by executor. Returns once the factorization is done.
 *
 * Every admissible leaf of the factors is compressed once, by partialACA with the epsilon of its LowRankMatrix, followed by
 * SVD_recompression if recompress is true.
 */
template <typename Executor, typename CoefficientPrecision, typename CoordinatePrecision>
void lazy_lu_factorization(Executor &executor, HMatrix<CoefficientPrecision, CoordinatePrecision> &hmatrix, bool recompress = false) {
    if (!hmatrix.is_block_tree_consistent()) {
        Logger::get_instance().log(LogLevel::ERROR, "lazy_lu_factorization is only implemented for consistent block tree."); // LCOV_EXCL_LINE
    }
    LazyLUDriver<Executor, CoefficientPrecision, CoordinatePrecision> driver(executor, recompress);
    driver.lu(hmatrix);
    executor.wait();
}

} // namespace htool

#endif
