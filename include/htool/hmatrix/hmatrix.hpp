#ifndef HTOOL_HMATRIX_HPP
#define HTOOL_HMATRIX_HPP

#if defined(_OPENMP)
#    include <omp.h>
#endif
#include "../basic_types/tree.hpp"                          // for Tree...
#include "../clustering/cluster_node.hpp"                   // for is_c...
#include "../matrix/matrix.hpp"                             // for Matrix
#include "../misc/logger.hpp"                               // for Logger
#include "../misc/misc.hpp"                                 // for unde...
#include "./interfaces/virtual_admissibility_condition.hpp" // for Virt...
#include "./interfaces/virtual_generator.hpp"               // for Virt...
#include "./interfaces/virtual_lrmat_generator.hpp"         // for Virt...
#include "hmatrix_tree_data.hpp"                            // for HMat...
#include "lrmat/lrmat.hpp"                                  // for LowR...
#include <algorithm>                                        // for min
#include <memory>                                           // for make...
#include <queue>                                            // for queue
#include <stack>                                            // for stack
#include <string>                                           // for basi...
#include <utility>                                          // for pair
#include <vector>                                           // for vector

namespace htool {

// Class
template <typename CoefficientPrecision, typename CoordinatePrecision = underlying_type<CoefficientPrecision>>
class HMatrix : public TreeNode<HMatrix<CoefficientPrecision, CoordinatePrecision>, HMatrixTreeData<CoefficientPrecision, CoordinatePrecision>> {
  public:
    enum class StorageType {
        Dense,
        LowRank,
        Hierarchical
    };

  private:
    // Data members
    const Cluster<CoordinatePrecision> *m_target_cluster, *m_source_cluster; // child's clusters are non owning
    char m_symmetry{'N'};
    char m_UPLO{'N'};

    std::unique_ptr<Matrix<CoefficientPrecision>> m_dense_data{nullptr};
    std::unique_ptr<LowRankMatrix<CoefficientPrecision, CoordinatePrecision>> m_low_rank_data{nullptr};

    //
    char m_symmetry_type_for_leaves{'N'};
    char m_UPLO_for_leaves{'N'};

    StorageType m_storage_type{StorageType::Hierarchical};

  public:
    // Root constructor
    HMatrix(const Cluster<CoordinatePrecision> &target_cluster, const Cluster<CoordinatePrecision> &source_cluster) : TreeNode<HMatrix, HMatrixTreeData<CoefficientPrecision, CoordinatePrecision>>(), m_target_cluster(&target_cluster), m_source_cluster(&source_cluster) {
    }

    // Child constructor
    HMatrix(const HMatrix &parent, const Cluster<CoordinatePrecision> *target_cluster, const Cluster<CoordinatePrecision> *source_cluster) : TreeNode<HMatrix, HMatrixTreeData<CoefficientPrecision, CoordinatePrecision>>(parent), m_target_cluster(target_cluster), m_source_cluster(source_cluster) {}

    HMatrix(const HMatrix &rhs) : TreeNode<HMatrix<CoefficientPrecision, CoordinatePrecision>, HMatrixTreeData<CoefficientPrecision, CoordinatePrecision>>(rhs), m_target_cluster(rhs.m_target_cluster), m_source_cluster(rhs.m_source_cluster), m_symmetry(rhs.m_symmetry), m_UPLO(rhs.m_UPLO), m_symmetry_type_for_leaves(rhs.m_symmetry_type_for_leaves), m_UPLO_for_leaves(rhs.m_UPLO_for_leaves), m_storage_type(rhs.m_storage_type) {
        if (m_target_cluster->is_root() or is_cluster_on_partition(*m_target_cluster)) {
            Logger::get_instance().log(LogLevel::INFO, "Deep copy of HMatrix");
        }
        this->m_depth     = rhs.m_depth;
        this->m_is_root   = rhs.m_is_root;
        this->m_tree_data = std::make_shared<HMatrixTreeData<CoefficientPrecision, CoordinatePrecision>>(*rhs.m_tree_data);
        this->m_children.clear();
        for (auto &child : rhs.m_children) {
            this->m_children.emplace_back(std::make_unique<HMatrix<CoefficientPrecision, CoordinatePrecision>>(*child));
        }
        if (rhs.m_dense_data) {
            m_dense_data = std::make_unique<Matrix<CoefficientPrecision>>(*rhs.m_dense_data);
        }
        if (rhs.m_low_rank_data) {
            m_low_rank_data = std::make_unique<LowRankMatrix<CoefficientPrecision, CoordinatePrecision>>(*rhs.m_low_rank_data);
        }
    }
    HMatrix &operator=(const HMatrix &rhs) {
        Logger::get_instance().log(LogLevel::INFO, "Deep copy of HMatrix");
        if (&rhs == this) {
            return *this;
        }
        this->m_depth     = rhs.m_depth;
        this->m_is_root   = rhs.m_is_root;
        this->m_tree_data = std::make_shared<HMatrixTreeData<CoefficientPrecision, CoordinatePrecision>>(*rhs.m_tree_data);
        this->m_children.clear();
        for (auto &child : rhs.m_children) {
            this->m_children.emplace_back(std::make_unique<HMatrix<CoefficientPrecision, CoordinatePrecision>>(*child));
        }
        m_target_cluster           = rhs.m_target_cluster;
        m_source_cluster           = rhs.m_source_cluster;
        m_symmetry                 = rhs.m_symmetry;
        m_UPLO                     = rhs.m_UPLO;
        m_storage_type             = rhs.m_storage_type;
        m_symmetry_type_for_leaves = rhs.m_symmetry_type_for_leaves;
        m_UPLO_for_leaves          = rhs.m_UPLO_for_leaves;

        if (rhs.m_dense_data) {
            m_dense_data = std::make_unique<Matrix<CoefficientPrecision>>(*rhs.m_dense_data);
        }
        if (rhs.m_low_rank_data) {
            m_low_rank_data = std::make_unique<LowRankMatrix<CoefficientPrecision, CoordinatePrecision>>(*rhs.m_low_rank_data);
        }
        return *this;
    }

    HMatrix(HMatrix &&) noexcept            = default;
    HMatrix &operator=(HMatrix &&) noexcept = default;
    virtual ~HMatrix()                      = default;

    // HMatrix getters
    const Cluster<CoordinatePrecision> &get_target_cluster() const { return *m_target_cluster; }
    const Cluster<CoordinatePrecision> &get_source_cluster() const { return *m_source_cluster; }
    int nb_cols() const { return m_source_cluster->get_size(); }
    int nb_rows() const { return m_target_cluster->get_size(); }
    htool::underlying_type<CoefficientPrecision> get_epsilon() const { return this->m_tree_data->m_epsilon; }

    HMatrix<CoefficientPrecision, CoordinatePrecision> *get_child_or_this(const Cluster<CoordinatePrecision> &required_target_cluster, const Cluster<CoordinatePrecision> &required_source_cluster) {
        if (*m_target_cluster == required_target_cluster and *m_source_cluster == required_source_cluster) {
            return this;
        }
        for (auto &child : this->m_children) {
            if (child->get_target_cluster() == required_target_cluster and child->get_source_cluster() == required_source_cluster) {
                return child.get();
            }
        }
        return nullptr;
    }

    const HMatrix<CoefficientPrecision, CoordinatePrecision> *get_child_or_this(const Cluster<CoordinatePrecision> &required_target_cluster, const Cluster<CoordinatePrecision> &required_source_cluster) const {
        if (*m_target_cluster == required_target_cluster and *m_source_cluster == required_source_cluster) {
            return this;
        }
        for (auto &child : this->m_children) {
            if (child->get_target_cluster() == required_target_cluster and child->get_source_cluster() == required_source_cluster) {
                return child.get();
            }
        }
        return nullptr;
    }

    int get_rank() const {
        return m_storage_type == StorageType::LowRank ? m_low_rank_data->rank_of() : -1;
    }

    const Matrix<CoefficientPrecision> *get_dense_data() const { return m_dense_data.get(); }
    Matrix<CoefficientPrecision> *get_dense_data() { return m_dense_data.get(); }
    const LowRankMatrix<CoefficientPrecision, CoordinatePrecision> *get_low_rank_data() const { return m_low_rank_data.get(); }
    LowRankMatrix<CoefficientPrecision, CoordinatePrecision> *get_low_rank_data() { return m_low_rank_data.get(); }
    char get_symmetry() const { return m_symmetry; }
    char get_UPLO() const { return m_UPLO; }
    const HMatrixTreeData<CoefficientPrecision, CoordinatePrecision> *get_hmatrix_tree_data() const { return this->m_tree_data.get(); }
    const HMatrix<CoefficientPrecision> *get_sub_hmatrix(const Cluster<CoordinatePrecision> &target_cluster, const Cluster<CoordinatePrecision> &source_cluster) const {
        std::queue<const HMatrix<CoefficientPrecision> *> hmatrix_queue;
        hmatrix_queue.push(this);

        while (!hmatrix_queue.empty()) {
            const HMatrix<CoefficientPrecision> *current_hmatrix = hmatrix_queue.front();
            hmatrix_queue.pop();

            if (target_cluster == current_hmatrix->get_target_cluster() && source_cluster == current_hmatrix->get_source_cluster()) {
                return current_hmatrix;
            }

            const auto &children = current_hmatrix->get_children();
            for (auto &child : children) {
                hmatrix_queue.push(child.get());
            }
        }
        return nullptr;
    }
    HMatrix<CoefficientPrecision> *get_sub_hmatrix(const Cluster<CoordinatePrecision> &target_cluster, const Cluster<CoordinatePrecision> &source_cluster) {
        std::queue<HMatrix<CoefficientPrecision> *> hmatrix_queue;
        hmatrix_queue.push(this);

        while (!hmatrix_queue.empty()) {
            HMatrix<CoefficientPrecision> *current_hmatrix = hmatrix_queue.front();
            hmatrix_queue.pop();

            if (target_cluster == current_hmatrix->get_target_cluster() && source_cluster == current_hmatrix->get_source_cluster()) {
                return current_hmatrix;
            }

            auto &children = current_hmatrix->get_children();
            for (auto &child : children) {
                hmatrix_queue.push(child.get());
            }
        }
        return nullptr;
    }
    StorageType get_storage_type() const { return m_storage_type; }

    // HMatrix node setters
    void set_symmetry(char symmetry) { m_symmetry = symmetry; }
    void set_UPLO(char UPLO) { m_UPLO = UPLO; }
    void set_symmetry_for_leaves(char symmetry) { m_symmetry_type_for_leaves = symmetry; }
    void set_UPLO_for_leaves(char UPLO) { m_UPLO_for_leaves = UPLO; }
    void set_target_cluster(const Cluster<CoordinatePrecision> *new_target_cluster) { m_target_cluster = new_target_cluster; }

    // Test properties
    bool is_dense() const { return m_storage_type == StorageType::Dense; }
    bool is_low_rank() const { return m_storage_type == StorageType::LowRank; }
    bool is_hierarchical() const { return m_storage_type == StorageType::Hierarchical; }

    // HMatrix Tree setters
    void set_eta(CoordinatePrecision eta) { this->m_tree_data->m_eta = eta; }
    void set_epsilon(underlying_type<CoefficientPrecision> epsilon) { this->m_tree_data->m_epsilon = epsilon; }
    void set_low_rank_generator(std::shared_ptr<VirtualInternalLowRankGenerator<CoefficientPrecision, CoordinatePrecision>> ptr) { this->m_tree_data->m_low_rank_generator = ptr; }
    void set_admissibility_condition(std::shared_ptr<VirtualAdmissibilityCondition<CoordinatePrecision>> ptr) { this->m_tree_data->m_admissibility_condition = ptr; }
    void set_minimal_target_depth(unsigned int minimal_target_depth) { this->m_tree_data->m_minimal_target_depth = minimal_target_depth; }
    void set_minimal_source_depth(unsigned int minimal_source_depth) { this->m_tree_data->m_minimal_source_depth = minimal_source_depth; }
    void set_block_tree_consistency(bool consistency) {
        this->m_tree_data->m_is_block_tree_consistent = consistency;
    }

    // HMatrix Tree setters
    char get_symmetry_for_leaves() const { return m_symmetry_type_for_leaves; }
    char get_UPLO_for_leaves() const { return m_UPLO_for_leaves; }
    bool is_block_tree_consistent() const { return this->m_tree_data->m_is_block_tree_consistent; }

    // Data computation
    void compute_dense_data(const VirtualInternalGenerator<CoefficientPrecision> &generator) {
        m_dense_data = std::make_unique<Matrix<CoefficientPrecision>>(m_target_cluster->get_size(), m_source_cluster->get_size());
        generator.copy_submatrix(m_target_cluster->get_size(), m_source_cluster->get_size(), m_target_cluster->get_offset(), m_source_cluster->get_offset(), m_dense_data->data());
        m_storage_type = StorageType::Dense;
    }

    void compute_low_rank_data(const VirtualInternalGenerator<CoefficientPrecision> &generator, const VirtualInternalLowRankGenerator<CoefficientPrecision, CoordinatePrecision> &low_rank_generator, int reqrank, underlying_type<CoefficientPrecision> epsilon) {
        m_low_rank_data = std::make_unique<LowRankMatrix<CoefficientPrecision, CoordinatePrecision>>(generator, low_rank_generator, *m_target_cluster, *m_source_cluster, reqrank, epsilon);
        m_storage_type  = StorageType::LowRank;
    }
    void clear_low_rank_data() { m_low_rank_data.reset(); }

    void set_dense_data(std::unique_ptr<Matrix<CoefficientPrecision>> dense_matrix_ptr) {
        this->delete_children();
        m_dense_data   = std::move(dense_matrix_ptr);
        m_storage_type = StorageType::Dense;
    }
};

template <typename CoefficientPrecision, typename CoordinatePrecision = underlying_type<CoefficientPrecision>>
std::pair<std::vector<const HMatrix<CoefficientPrecision, CoordinatePrecision> *>, std::vector<const HMatrix<CoefficientPrecision, CoordinatePrecision> *>> get_leaves_from(const HMatrix<CoefficientPrecision, CoordinatePrecision> &hmatrix) {
    std::pair<std::vector<const HMatrix<CoefficientPrecision, CoordinatePrecision> *>, std::vector<const HMatrix<CoefficientPrecision, CoordinatePrecision> *>> result;
    auto &leaves              = result.first;
    auto &leaves_for_symmetry = result.second;
    std::stack<std::pair<const HMatrix<CoefficientPrecision, CoordinatePrecision> *, bool>> hmatrix_stack;
    hmatrix_stack.push(std::pair<const HMatrix<CoefficientPrecision, CoordinatePrecision> *, bool>(&hmatrix, hmatrix.get_symmetry() != 'N'));

    while (!hmatrix_stack.empty()) {
        auto &current_element = hmatrix_stack.top();
        hmatrix_stack.pop();
        const HMatrix<CoefficientPrecision, CoordinatePrecision> *current_hmatrix = current_element.first;
        bool has_symmetric_ancestor                                               = current_element.second;

        if (current_hmatrix->is_leaf()) {
            leaves.push_back(current_hmatrix);

            if (has_symmetric_ancestor && current_hmatrix->get_target_cluster().get_offset() != current_hmatrix->get_source_cluster().get_offset()) {
                leaves_for_symmetry.push_back(current_hmatrix);
            }
        }

        for (const auto &child : current_hmatrix->get_children()) {
            hmatrix_stack.push(std::pair<const HMatrix<CoefficientPrecision, CoordinatePrecision> *, bool>(child.get(), current_hmatrix->get_symmetry() != 'N' || has_symmetric_ancestor));
        }
    }
    return result;
}

template <typename CoefficientPrecision, typename CoordinatePrecision = underlying_type<CoefficientPrecision>>
void copy_to_dense(const HMatrix<CoefficientPrecision, CoordinatePrecision> &hmatrix, CoefficientPrecision *ptr) {

    int target_offset = hmatrix.get_target_cluster().get_offset();
    int source_offset = hmatrix.get_source_cluster().get_offset();
    int target_size   = hmatrix.get_target_cluster().get_size();

    std::vector<const HMatrix<CoefficientPrecision, CoordinatePrecision> *> leaves;
    std::vector<const HMatrix<CoefficientPrecision, CoordinatePrecision> *> leaves_for_symmetry;
    std::tie(leaves, leaves_for_symmetry) = get_leaves_from(hmatrix); // C++17 structured binding

    for (auto leaf : leaves) {
        int local_nr   = leaf->get_target_cluster().get_size();
        int local_nc   = leaf->get_source_cluster().get_size();
        int row_offset = leaf->get_target_cluster().get_offset() - target_offset;
        int col_offset = leaf->get_source_cluster().get_offset() - source_offset;
        if (leaf->is_dense()) {
            for (int k = 0; k < local_nc; k++) {
                for (int j = 0; j < local_nr; j++) {
                    ptr[j + row_offset + (k + col_offset) * target_size] = (*leaf->get_dense_data())(j, k);
                }
            }
        } else {

            Matrix<CoefficientPrecision> low_rank_to_dense(local_nr, local_nc);
            leaf->get_low_rank_data()->copy_to_dense(low_rank_to_dense.data());
            for (int k = 0; k < local_nc; k++) {
                for (int j = 0; j < local_nr; j++) {
                    ptr[j + row_offset + (k + col_offset) * target_size] = low_rank_to_dense(j, k);
                }
            }
        }
    }
}

template <typename CoefficientPrecision, typename CoordinatePrecision = underlying_type<CoefficientPrecision>>
void copy_to_dense_in_user_numbering(const HMatrix<CoefficientPrecision, CoordinatePrecision> &hmatrix, CoefficientPrecision *ptr) {

    const auto &target_cluster = hmatrix.get_target_cluster();
    const auto &source_cluster = hmatrix.get_source_cluster();
    if (!target_cluster.is_root() && !is_cluster_on_partition(target_cluster)) {
        htool::Logger::get_instance().log(LogLevel::ERROR, "Target cluster is neither root nor local, permutation is not stable and copy_to_dense_in_user_numbering cannot be used."); // LCOV_EXCL_LINE
    }

    if (!source_cluster.is_root() && !is_cluster_on_partition(source_cluster)) {
        htool::Logger::get_instance().log(LogLevel::ERROR, "Source cluster is neither root nor local, permutation is not stable and copy_to_dense_in_user_numbering cannot be used."); // LCOV_EXCL_LINE
    }

    if (is_cluster_on_partition(target_cluster) && !(target_cluster.is_permutation_local())) {
        htool::Logger::get_instance().log(LogLevel::ERROR, "Target cluster is local, but permutation is not local, copy_to_dense_in_user_numbering cannot be used"); // LCOV_EXCL_LINE}
    }

    if (is_cluster_on_partition(source_cluster) && !(source_cluster.is_permutation_local())) {
        htool::Logger::get_instance().log(LogLevel::ERROR, "Source cluster is local, but permutation is not local, copy_to_dense_in_user_numbering cannot be used"); // LCOV_EXCL_LINE}
    }
    int target_offset              = target_cluster.get_offset();
    int source_offset              = source_cluster.get_offset();
    int target_size                = target_cluster.get_size();
    const auto &target_permutation = target_cluster.get_permutation();
    const auto &source_permutation = source_cluster.get_permutation();
    std::vector<const HMatrix<CoefficientPrecision, CoordinatePrecision> *> leaves;
    std::vector<const HMatrix<CoefficientPrecision, CoordinatePrecision> *> leaves_for_symmetry;
    std::tie(leaves, leaves_for_symmetry) = get_leaves_from(hmatrix); // C++17 structured binding

    for (auto leaf : leaves) {
        int local_nr   = leaf->get_target_cluster().get_size();
        int local_nc   = leaf->get_source_cluster().get_size();
        int row_offset = leaf->get_target_cluster().get_offset();
        int col_offset = leaf->get_source_cluster().get_offset();

        if (leaf->is_dense()) {
            for (int k = 0; k < local_nc; k++) {
                for (int j = 0; j < local_nr; j++) {
                    ptr[target_permutation[j + row_offset] - target_offset + (source_permutation[k + col_offset] - source_offset) * target_size] = (*leaf->get_dense_data())(j, k);
                }
            }
        } else {

            Matrix<CoefficientPrecision> low_rank_to_dense(local_nr, local_nc);
            leaf->get_low_rank_data()->copy_to_dense(low_rank_to_dense.data());
            for (int k = 0; k < local_nc; k++) {
                for (int j = 0; j < local_nr; j++) {
                    ptr[target_permutation[j + row_offset] - target_offset + (source_permutation[k + col_offset] - source_offset) * target_size] = low_rank_to_dense(j, k);
                }
            }
        }
    }
    if (hmatrix.get_symmetry_for_leaves() != 'N') {
        for (auto leaf : leaves_for_symmetry) {
            int local_nr   = leaf->get_target_cluster().get_size();
            int local_nc   = leaf->get_source_cluster().get_size();
            int row_offset = leaf->get_target_cluster().get_offset();
            int col_offset = leaf->get_source_cluster().get_offset();

            for (int k = 0; k < local_nc; k++) {
                for (int j = 0; j < local_nr; j++) {
                    ptr[source_permutation[k + col_offset] - target_offset + (target_permutation[j + row_offset] - source_offset) * target_size] = hmatrix.get_symmetry_for_leaves() == 'S' ? ptr[target_permutation[j + row_offset] - target_offset + (source_permutation[k + col_offset] - source_offset) * target_size] : conj_if_complex(ptr[target_permutation[j + row_offset] - target_offset + (source_permutation[k + col_offset] - source_offset) * target_size]);
                }
            }
        }
    }
}

template <typename CoefficientPrecision, typename CoordinatePrecision = underlying_type<CoefficientPrecision>>
void copy_diagonal(const HMatrix<CoefficientPrecision, CoordinatePrecision> &hmatrix, CoefficientPrecision *ptr) {
    if (hmatrix.get_target_cluster().get_offset() != hmatrix.get_source_cluster().get_offset() || hmatrix.get_target_cluster().get_size() != hmatrix.get_source_cluster().get_size()) {
        htool::Logger::get_instance().log(LogLevel::ERROR, "Matrix is not square a priori, get_local_diagonal cannot be used"); // LCOV_EXCL_LINE
        // throw std::logic_error("[Htool error] Matrix is not square a priori, get_local_diagonal cannot be used");                       // LCOV_EXCL_LINE
    }

    int target_offset = hmatrix.get_target_cluster().get_offset();
    int source_offset = hmatrix.get_source_cluster().get_offset();
    std::vector<const HMatrix<CoefficientPrecision, CoordinatePrecision> *> leaves;
    std::vector<const HMatrix<CoefficientPrecision, CoordinatePrecision> *> leaves_for_symmetry;
    std::tie(leaves, leaves_for_symmetry) = get_leaves_from(hmatrix); // C++17 structured binding

    for (auto leaf : leaves) {
        int local_nr = leaf->get_target_cluster().get_size();
        int local_nc = leaf->get_source_cluster().get_size();
        int offset_i = leaf->get_target_cluster().get_offset() - target_offset;
        int offset_j = leaf->get_source_cluster().get_offset() - source_offset;
        if (leaf->is_dense() && offset_i == offset_j) {
            for (int k = 0; k < std::min(local_nr, local_nc); k++) {
                ptr[k + offset_i] = (*leaf->get_dense_data())(k, k);
            }
        } else if (leaf->is_low_rank() && offset_i == offset_j) { // pretty rare...
            Matrix<CoefficientPrecision> low_rank_to_dense(local_nc, local_nr);
            leaf->get_low_rank_data()->copy_to_dense(low_rank_to_dense.data());
            for (int k = 0; k < std::min(local_nr, local_nc); k++) {
                ptr[k + offset_i] = (low_rank_to_dense)(k, k);
            }
        }
    }
}

template <typename CoefficientPrecision, typename CoordinatePrecision = underlying_type<CoefficientPrecision>>
void copy_diagonal_in_user_numbering(const HMatrix<CoefficientPrecision, CoordinatePrecision> &hmatrix, CoefficientPrecision *ptr) {
    if (hmatrix.get_target_cluster().get_offset() != hmatrix.get_source_cluster().get_offset() || hmatrix.get_target_cluster().get_size() != hmatrix.get_source_cluster().get_size()) {
        htool::Logger::get_instance().log(LogLevel::ERROR, "Matrix is not square a priori, get_local_diagonal cannot be used"); // LCOV_EXCL_LINE
    }

    if (!(hmatrix.get_target_cluster().is_root() && hmatrix.get_source_cluster().is_root()) && !(is_cluster_on_partition(hmatrix.get_target_cluster()) && is_cluster_on_partition(hmatrix.get_source_cluster()))) {
        htool::Logger::get_instance().log(LogLevel::ERROR, "Clusters are neither root nor local, permutations are not stable and copy_diagonal_in_user_numbering cannot be used."); // LCOV_EXCL_LINE
    }

    if ((is_cluster_on_partition(hmatrix.get_target_cluster()) && is_cluster_on_partition(hmatrix.get_source_cluster())) && !(hmatrix.get_target_cluster().is_permutation_local() && hmatrix.get_source_cluster().is_permutation_local())) {
        htool::Logger::get_instance().log(LogLevel::ERROR, "Clusters are local, but permutations are not local, copy_diagonal_in_user_numbering cannot be used"); // LCOV_EXCL_LINE}
    }

    const auto &permutation = hmatrix.get_target_cluster().get_permutation();
    int target_offset       = hmatrix.get_target_cluster().get_offset();
    // int source_offset       = hmatrix.get_source_cluster().get_offset();
    std::vector<const HMatrix<CoefficientPrecision, CoordinatePrecision> *> leaves;
    std::vector<const HMatrix<CoefficientPrecision, CoordinatePrecision> *> leaves_for_symmetry;
    std::tie(leaves, leaves_for_symmetry) = get_leaves_from(hmatrix); // C++17 structured binding

    for (auto leaf : leaves) {
        int local_nr = leaf->get_target_cluster().get_size();
        int local_nc = leaf->get_source_cluster().get_size();
        int offset_i = leaf->get_target_cluster().get_offset();
        int offset_j = leaf->get_source_cluster().get_offset();
        if (leaf->is_dense() && offset_i == offset_j) {
            for (int k = 0; k < std::min(local_nr, local_nc); k++) {
                ptr[permutation[k + offset_i] - target_offset] = (*leaf->get_dense_data())(k, k);
            }
        } else if (leaf->is_low_rank() && offset_i == offset_j) { // pretty rare...
            Matrix<CoefficientPrecision> low_rank_to_dense(local_nc, local_nr);
            leaf->get_low_rank_data()->copy_to_dense(low_rank_to_dense.data());
            for (int k = 0; k < std::min(local_nr, local_nc); k++) {
                ptr[permutation[k + offset_i] - target_offset] = (low_rank_to_dense)(k, k);
            }
        }
    }
}

} // namespace htool
#endif
