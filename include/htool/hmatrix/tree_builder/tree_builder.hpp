#ifndef HTOOL_HMATRIX_TREE_BUILDER_HPP
#define HTOOL_HMATRIX_TREE_BUILDER_HPP

#include "../../misc/logger.hpp"
#include "../hmatrix.hpp"
#include "../interfaces/virtual_dense_blocks_generator.hpp"
#include "../lrmat/sympartialACA.hpp"
#include <chrono>

namespace htool {

template <typename CoefficientPrecision, typename CoordinatePrecision>
class HMatrixTreeBuilder {
  private:
    class ZeroGenerator : public VirtualGenerator<CoefficientPrecision> {
        void copy_submatrix(int M, int N, int, int, CoefficientPrecision *ptr) const override {
            std::fill_n(ptr, M * N, CoefficientPrecision(0));
        }
    };

    using HMatrixType = HMatrix<CoefficientPrecision, CoordinatePrecision>;
    using ClusterType = Cluster<CoordinatePrecision>;

    // Parameters
    const ClusterType &m_target_root_cluster, &m_source_root_cluster;
    underlying_type<CoefficientPrecision> m_epsilon{1e-6};
    CoordinatePrecision m_eta{10};
    int m_maxblocksize{1000000};
    int m_minsourcedepth{0};
    int m_mintargetdepth{0};
    int m_reqrank{-1};
    int m_target_partition_number{-1};

    char m_symmetry_type{'N'};
    char m_UPLO_type{'N'};
    int m_partition_number_for_symmetry{-1};

    // Views
    mutable std::vector<HMatrixType *> m_admissible_tasks{};
    mutable std::vector<HMatrixType *> m_dense_tasks{};

    // Information
    mutable int m_false_positive{0};

    // Strategies
    std::shared_ptr<VirtualLowRankGenerator<CoefficientPrecision, CoordinatePrecision>> m_low_rank_generator;
    std::shared_ptr<VirtualAdmissibilityCondition<CoordinatePrecision>> m_admissibility_condition;
    std::shared_ptr<VirtualDenseBlocksGenerator<CoefficientPrecision>> m_dense_blocks_generator;

    // Internal methods
    bool build_block_tree(HMatrixType *current_hmatrix) const;
    void reset_root_of_block_tree(HMatrixType &) const;
    void compute_blocks(const VirtualGenerator<CoefficientPrecision> &generator) const;

    // Tests
    bool is_target_cluster_in_target_partition(const ClusterType &cluster) const {
        return (m_target_partition_number == -1) ? true : (m_target_partition_number == cluster.get_rank());
    }
    bool is_removed_by_symmetry(const ClusterType &target_cluster, const ClusterType &source_cluster) const {
        return (m_symmetry_type != 'N')
               && ((m_UPLO_type == 'U'
                    && target_cluster.get_offset() >= (source_cluster.get_offset() + source_cluster.get_size())
                    && ((m_partition_number_for_symmetry == -1)
                        || (source_cluster.get_offset() >= m_source_root_cluster.get_clusters_on_partition()[m_partition_number_for_symmetry]->get_offset() and m_target_root_cluster.get_clusters_on_partition()[m_partition_number_for_symmetry]->get_offset() <= target_cluster.get_offset() && target_cluster.get_offset() + target_cluster.get_size() <= m_target_root_cluster.get_clusters_on_partition()[m_partition_number_for_symmetry]->get_offset() + m_target_root_cluster.get_clusters_on_partition()[m_partition_number_for_symmetry]->get_size()))
                    // && ((m_target_partition_number != -1)
                    //     || source_cluster.get_offset() >= target_cluster.get_offset())
                    )
                   || (m_UPLO_type == 'L'
                       && source_cluster.get_offset() >= (target_cluster.get_offset() + target_cluster.get_size())
                       && ((m_partition_number_for_symmetry == -1)
                           || (source_cluster.get_offset() < m_source_root_cluster.get_clusters_on_partition()[m_partition_number_for_symmetry]->get_offset() + m_source_root_cluster.get_clusters_on_partition()[m_partition_number_for_symmetry]->get_size() && m_target_root_cluster.get_clusters_on_partition()[m_partition_number_for_symmetry]->get_offset() <= target_cluster.get_offset() && target_cluster.get_offset() + target_cluster.get_size() <= m_target_root_cluster.get_clusters_on_partition()[m_partition_number_for_symmetry]->get_offset() + m_target_root_cluster.get_clusters_on_partition()[m_partition_number_for_symmetry]->get_size()))
                       //    && ((m_target_partition_number != -1)
                       //    || source_cluster.get_offset() < target_cluster.get_offset() + target_cluster.get_size())
                       ));
    }
    bool is_block_diagonal(const HMatrixType &hmatrix) const {
        bool is_there_a_target_partition = (m_target_partition_number != -1);
        const auto &target_cluster       = hmatrix.get_target_cluster();
        const auto &source_cluster       = hmatrix.get_source_cluster();

        return (is_there_a_target_partition
                && (target_cluster == *m_target_root_cluster.get_clusters_on_partition()[m_target_partition_number])
                && source_cluster == *m_source_root_cluster.get_clusters_on_partition()[m_target_partition_number])
               || (!is_there_a_target_partition
                   && target_cluster == m_target_root_cluster
                   && target_cluster == source_cluster);
    }

    void set_hmatrix_symmetry(HMatrixType &hmatrix) const {
        if (m_symmetry_type != 'N'
            && hmatrix.get_target_cluster().get_offset() == hmatrix.get_source_cluster().get_offset()
            && hmatrix.get_target_cluster().get_size() == hmatrix.get_source_cluster().get_size()) {
            hmatrix.set_symmetry(m_symmetry_type);
            hmatrix.set_UPLO(m_UPLO_type);
        }
    }

  public:
    explicit HMatrixTreeBuilder(const ClusterType &root_cluster_tree_target, const ClusterType &root_source_cluster_tree, underlying_type<CoefficientPrecision> epsilon, CoordinatePrecision eta, char symmetry, char UPLO, int reqrank, int target_partition_number, int partition_number_for_symmetry) : m_target_root_cluster(root_cluster_tree_target), m_source_root_cluster(root_source_cluster_tree), m_epsilon(epsilon), m_eta(eta), m_reqrank(reqrank), m_target_partition_number(target_partition_number), m_symmetry_type(symmetry), m_UPLO_type(UPLO), m_partition_number_for_symmetry(partition_number_for_symmetry), m_low_rank_generator(std::make_shared<sympartialACA<CoefficientPrecision, CoordinatePrecision>>()), m_admissibility_condition(std::make_shared<RjasanowSteinbach<CoordinatePrecision>>()) {
        if (!((m_symmetry_type == 'N' || m_symmetry_type == 'H' || m_symmetry_type == 'S')
              && (m_UPLO_type == 'N' || m_UPLO_type == 'L' || m_UPLO_type == 'U')
              && ((m_symmetry_type == 'N' && m_UPLO_type == 'N') || (m_symmetry_type != 'N' && m_UPLO_type != 'N'))
              && ((m_symmetry_type == 'H' && is_complex<CoefficientPrecision>()) || m_symmetry_type != 'H'))) {
            std::string error_message = "[Htool error] Invalid arguments to create HMatrix: m_symmetry_type=";
            error_message.push_back(m_symmetry_type);
            error_message += " and m_UPLO_type=";
            error_message.push_back(m_UPLO_type);
            htool::Logger::get_instance().log(LogLevel::ERROR, error_message); // LCOV_EXCL_LINE
        }
        if (target_partition_number != -1 && target_partition_number >= m_target_root_cluster.get_clusters_on_partition().size()) {
            htool::Logger::get_instance().log(LogLevel::ERROR, "Target partition number cannot exceed number of partitions"); // LCOV_EXCL_LINE
        }
        if (partition_number_for_symmetry != -1 && partition_number_for_symmetry >= m_target_root_cluster.get_clusters_on_partition().size()) {
            htool::Logger::get_instance().log(LogLevel::ERROR, "Partition number for symmetry cannot exceed number of partitions"); // LCOV_EXCL_LINE
        }
    }

    HMatrixTreeBuilder(const HMatrixTreeBuilder &)                = delete;
    HMatrixTreeBuilder &operator=(const HMatrixTreeBuilder &)     = delete;
    HMatrixTreeBuilder(HMatrixTreeBuilder &&) noexcept            = default;
    HMatrixTreeBuilder &operator=(HMatrixTreeBuilder &&) noexcept = default;
    virtual ~HMatrixTreeBuilder()                                 = default;

    // Build
    HMatrixType build(const VirtualGenerator<CoefficientPrecision> &generator) const;

    // Setters
    void set_low_rank_generator(std::shared_ptr<VirtualLowRankGenerator<CoefficientPrecision, CoordinatePrecision>> ptr) { m_low_rank_generator = ptr; }
    void set_admissibility_condition(std::shared_ptr<VirtualAdmissibilityCondition<CoordinatePrecision>> ptr) { m_admissibility_condition = ptr; }
    void set_maximal_block_size(int maxblock_size) { m_maxblocksize = maxblock_size; }
    void set_minimal_source_depth(int minimal_source_depth) { m_minsourcedepth = minimal_source_depth; }
    void set_minimal_target_depth(int minimal_target_depth) { m_mintargetdepth = minimal_target_depth; }
    void set_dense_blocks_generator(std::shared_ptr<VirtualDenseBlocksGenerator<CoefficientPrecision>> dense_blocks_generator) { m_dense_blocks_generator = dense_blocks_generator; }
};

template <typename CoefficientPrecision, typename CoordinatePrecision>
HMatrix<CoefficientPrecision, CoordinatePrecision> HMatrixTreeBuilder<CoefficientPrecision, CoordinatePrecision>::build(const VirtualGenerator<CoefficientPrecision> &generator) const {
    // Create root hmatrix
    HMatrixType root_hmatrix(m_target_root_cluster, m_source_root_cluster);
    root_hmatrix.set_admissibility_condition(m_admissibility_condition);
    root_hmatrix.set_low_rank_generator(m_low_rank_generator);
    root_hmatrix.set_eta(m_eta);
    root_hmatrix.set_epsilon(m_epsilon);
    root_hmatrix.set_maximal_block_size(m_maxblocksize);
    root_hmatrix.set_minimal_target_depth(m_mintargetdepth);
    root_hmatrix.set_minimal_source_depth(m_minsourcedepth);

    // Build hierarchical block structure
    std::chrono::steady_clock::time_point start, end;
    start           = std::chrono::steady_clock::now();
    bool not_pushed = false;
    not_pushed      = build_block_tree(&root_hmatrix);

    if (not_pushed) {

        bool is_root_admissible = m_admissibility_condition->ComputeAdmissibility(root_hmatrix.get_target_cluster(), root_hmatrix.get_source_cluster(), m_eta);
        if (is_root_admissible) {
            m_admissible_tasks.emplace_back(&root_hmatrix);
        } else {
            m_dense_tasks.emplace_back(&root_hmatrix);
        }
    }
    reset_root_of_block_tree(root_hmatrix);

    set_hmatrix_symmetry(root_hmatrix);
    end = std::chrono::steady_clock::now();

    std::chrono::duration<double> block_tree_build_duration = end - start;

    // Compute leave's data
    start = std::chrono::steady_clock::now();
    compute_blocks(generator);
    end = std::chrono::steady_clock::now();

    std::chrono::duration<double> block_comptations_duration = end - start;

    // Set information
    root_hmatrix.get_hmatrix_tree_data()->m_information["Number_of_false_positive"] = NbrToStr(m_false_positive);
    root_hmatrix.get_hmatrix_tree_data()->m_timings["Block_tree_walltime"]          = block_tree_build_duration;
    root_hmatrix.get_hmatrix_tree_data()->m_timings["Blocks_computation_walltime"]  = block_comptations_duration;

    return root_hmatrix;
}

template <typename CoefficientPrecision, typename CoordinatePrecision>
bool HMatrixTreeBuilder<CoefficientPrecision, CoordinatePrecision>::build_block_tree(HMatrixType *current_hmatrix) const {
    const auto &target_cluster = current_hmatrix->get_target_cluster();
    const auto &source_cluster = current_hmatrix->get_source_cluster();
    std::size_t block_size     = std::size_t(target_cluster.get_size()) * std::size_t(source_cluster.get_size());
    // int target_partition_number = m_target_partition_number;
    // bool is_there_a_target_partition = (target_partition_number != -1);
    bool is_admissible = m_admissibility_condition->ComputeAdmissibility(target_cluster, source_cluster, m_eta);

    ///////////////////// Diagonal blocks
    // int rankWorld;
    // MPI_Comm_rank(MPI_COMM_WORLD, &rankWorld);
    // if (rankWorld == 0)
    //     std::cout << target_cluster.get_offset() << " " << target_cluster.get_size() << " " << source_cluster.get_offset() << " " << source_cluster.get_size() << " " << is_block_diagonal(*current_hmatrix) << " " << is_target_cluster_in_target_partition(target_cluster) << " " << target_cluster.get_rank() << " " << is_removed_by_symmetry(target_cluster, source_cluster) << "\n";
    // if (is_block_diagonal(*current_hmatrix)) {
    //     current_hmatrix->set_diagonal_hmatrix(current_hmatrix);
    // }

    ///////////////////// Recursion
    const auto &target_children = target_cluster.get_children();
    const auto &source_children = source_cluster.get_children();

    if (is_admissible
        && is_target_cluster_in_target_partition(target_cluster)
        && !is_removed_by_symmetry(target_cluster, source_cluster)
        && target_cluster.get_depth() >= m_mintargetdepth
        && source_cluster.get_depth() >= m_minsourcedepth
        && target_cluster.get_rank() >= 0
        && source_cluster.get_rank() >= 0) {
        m_admissible_tasks.push_back(current_hmatrix);
        return false;
    } else if (source_cluster.is_leaf()) {
        if (target_cluster.is_leaf()) {
            return true;
        } else {
            std::vector<bool> Blocks_not_pushed{};
            std::vector<HMatrixType *> child_blocks{};
            for (const auto &target_child : target_children) {
                if ((is_target_cluster_in_target_partition(*target_child) || target_cluster.get_rank() < 0) && !is_removed_by_symmetry(*target_child, source_cluster)) {
                    // child_blocks.emplace_back(new HMatrixType(target_child, source_cluster, current_hmatrix->m_depth + 1));
                    child_blocks.emplace_back(current_hmatrix->add_child(target_child.get(), &source_cluster));
                    set_hmatrix_symmetry(*child_blocks.back());
                    Blocks_not_pushed.push_back(build_block_tree(child_blocks.back()));
                }
            }

            // All sons are non admissible and not pushed to tasks -> the current block is not pushed
            if ((block_size <= m_maxblocksize)
                && std::all_of(Blocks_not_pushed.begin(), Blocks_not_pushed.end(), [](bool i) { return i; })
                && is_target_cluster_in_target_partition(target_cluster)
                && target_cluster.get_depth() >= m_mintargetdepth
                && source_cluster.get_depth() >= m_minsourcedepth
                && std::all_of(child_blocks.begin(), child_blocks.end(), [this](HMatrixType *block) {
                       //    return (block != block->get_diagonal_hmatrix());
                       return !is_block_diagonal(*block);
                   })) {
                child_blocks.clear();
                current_hmatrix->delete_children();
                return true;
            }
            // Some sons have been pushed, we cannot go higher. Every other sons are also pushed so that the current block is done
            else {
                for (int p = 0; p < Blocks_not_pushed.size(); p++) {
                    if (Blocks_not_pushed[p]) {
                        m_dense_tasks.push_back(child_blocks[p]);
                    }
                }
                return false;
            }
        }
    } else {
        if (target_cluster.is_leaf()) {
            std::vector<bool> Blocks_not_pushed{};
            std::vector<HMatrixType *> child_blocks{};
            for (const auto &source_child : source_children) {
                if (!is_removed_by_symmetry(target_cluster, *source_child)) {
                    child_blocks.emplace_back(current_hmatrix->add_child(&target_cluster, source_child.get()));
                    set_hmatrix_symmetry(*child_blocks.back());
                    Blocks_not_pushed.push_back(build_block_tree(child_blocks.back()));
                }
            }

            if ((block_size <= m_maxblocksize)
                && std::all_of(Blocks_not_pushed.begin(), Blocks_not_pushed.end(), [](bool i) { return i; })
                && is_target_cluster_in_target_partition(target_cluster)
                && target_cluster.get_depth() >= m_mintargetdepth
                && source_cluster.get_depth() >= m_minsourcedepth
                && std::all_of(child_blocks.begin(), child_blocks.end(), [this](HMatrixType *block) {
                       //    return (block != block->get_diagonal_hmatrix());
                       return !is_block_diagonal(*block);
                   })) {
                child_blocks.clear();
                current_hmatrix->delete_children();
                return true;
            } else {
                for (int p = 0; p < Blocks_not_pushed.size(); p++) {
                    if (Blocks_not_pushed[p]) {
                        m_dense_tasks.push_back(child_blocks[p]);
                    }
                }

                return false;
            }
        } else {
            if (target_cluster.get_rank() < 0 && source_cluster.get_rank() >= 0) {
                std::vector<bool> Blocks_not_pushed{};
                std::vector<HMatrixType *> child_blocks{};
                for (const auto &target_child : target_cluster.get_clusters_on_partition()) {
                    if ((is_target_cluster_in_target_partition(*target_child) || target_cluster.get_rank() < 0) && !is_removed_by_symmetry(*target_child, source_cluster)) {
                        child_blocks.emplace_back(current_hmatrix->add_child(target_child, &source_cluster));
                        set_hmatrix_symmetry(*child_blocks.back());
                        Blocks_not_pushed.push_back(build_block_tree(child_blocks.back()));
                    }
                }
                if ((block_size <= m_maxblocksize)
                    && std::all_of(Blocks_not_pushed.begin(), Blocks_not_pushed.end(), [](bool i) { return i; })
                    && is_target_cluster_in_target_partition(target_cluster)
                    && target_cluster.get_depth() >= m_mintargetdepth
                    && source_cluster.get_depth() >= m_minsourcedepth
                    && std::all_of(child_blocks.begin(), child_blocks.end(), [this](HMatrixType *block) {
                           //    return (block != block->get_diagonal_hmatrix());
                           return !is_block_diagonal(*block);
                       })) {
                    child_blocks.clear();
                    current_hmatrix->delete_children();
                    return true;
                } else {
                    for (int p = 0; p < Blocks_not_pushed.size(); p++) {
                        if (Blocks_not_pushed[p]) {
                            m_dense_tasks.push_back(child_blocks[p]);
                        }
                    }

                    return false;
                }
            } else if (source_cluster.get_rank() < 0 && target_cluster.get_rank() >= 0) {
                std::vector<bool> Blocks_not_pushed;
                std::vector<HMatrixType *> child_blocks{};
                for (const auto &source_child : source_cluster.get_clusters_on_partition()) {
                    if (!is_removed_by_symmetry(target_cluster, *source_child)) {
                        child_blocks.emplace_back(current_hmatrix->add_child(&target_cluster, source_child));
                        set_hmatrix_symmetry(*child_blocks.back());
                        Blocks_not_pushed.push_back(build_block_tree(child_blocks.back()));
                    }
                }
                if ((block_size <= m_maxblocksize)
                    && std::all_of(Blocks_not_pushed.begin(), Blocks_not_pushed.end(), [](bool i) { return i; })
                    && is_target_cluster_in_target_partition(target_cluster)
                    && target_cluster.get_depth() >= m_mintargetdepth
                    && source_cluster.get_depth() >= m_minsourcedepth
                    && std::all_of(child_blocks.begin(), child_blocks.end(), [this](HMatrixType *block) {
                           //    return (block != block->get_diagonal_hmatrix());
                           return !is_block_diagonal(*block);
                       })) {
                    child_blocks.clear();
                    current_hmatrix->delete_children();
                    return true;
                } else {
                    for (int p = 0; p < Blocks_not_pushed.size(); p++) {
                        if (Blocks_not_pushed[p]) {
                            m_dense_tasks.push_back(child_blocks[p]);
                        }
                    }

                    return false;
                }
            } else {
                std::vector<bool> Blocks_not_pushed;
                std::vector<HMatrixType *> child_blocks{};
                for (const auto &target_child : target_children) {
                    for (const auto &source_child : source_children) {
                        if ((is_target_cluster_in_target_partition(*target_child) || target_cluster.get_rank() < 0) && !is_removed_by_symmetry(*target_child, *source_child)) {
                            child_blocks.emplace_back(current_hmatrix->add_child(target_child.get(), source_child.get()));
                            set_hmatrix_symmetry(*child_blocks.back());
                            Blocks_not_pushed.push_back(build_block_tree(child_blocks.back()));
                        }
                    }
                }
                if ((block_size <= m_maxblocksize)
                    && std::all_of(Blocks_not_pushed.begin(), Blocks_not_pushed.end(), [](bool i) { return i; })
                    && is_target_cluster_in_target_partition(target_cluster)
                    && target_cluster.get_depth() >= m_mintargetdepth
                    && source_cluster.get_depth() >= m_minsourcedepth
                    && std::all_of(child_blocks.begin(), child_blocks.end(), [this](HMatrixType *block) {
                           //    return (block != block->get_diagonal_hmatrix());
                           return !is_block_diagonal(*block);
                       })) {
                    child_blocks.clear();
                    current_hmatrix->delete_children();
                    return true;
                } else {
                    for (int p = 0; p < Blocks_not_pushed.size(); p++) {
                        if (Blocks_not_pushed[p]) {
                            m_dense_tasks.push_back(child_blocks[p]);
                        }
                    }

                    return false;
                }
            }
        }
    }
}

template <typename CoefficientPrecision, typename CoordinatePrecision>
void HMatrixTreeBuilder<CoefficientPrecision, CoordinatePrecision>::reset_root_of_block_tree(HMatrixType &root_hmatrix) const {

    if (!is_target_cluster_in_target_partition(root_hmatrix.get_target_cluster())) {
        int target_partition_number = m_target_partition_number;
        std::stack<HMatrixType *> block_stack;
        block_stack.emplace(&root_hmatrix);

        std::vector<std::unique_ptr<HMatrixType>> new_root_children;

        while (!block_stack.empty()) {
            auto &current_hmatrix = block_stack.top();
            block_stack.pop();

            for (auto &child : current_hmatrix->get_children_with_ownership()) {
                if (child->get_target_cluster().get_rank() == target_partition_number) {
                    new_root_children.push_back(std::move(child));
                } else {

                    block_stack.push(child.get());
                }
            }
        }

        root_hmatrix.delete_children();
        root_hmatrix.assign_children(new_root_children);
        root_hmatrix.set_target_cluster(root_hmatrix.get_target_cluster().get_clusters_on_partition()[target_partition_number]);
        // m_storage.emplace_back(new HMatrixType(m_target_root_cluster.get_clusters_on_partition()[target_partition_number], &m_root_hmatrix->get_source_cluster(), 0));
        // m_root_hmatrix = m_storage.back().get();
        // for (const auto &new_child : new_root_children) {
        //     m_storage.back().get()->m_children.push_back(new_child);
        // }
    }
}

template <typename CoefficientPrecision, typename CoordinatePrecision>
void HMatrixTreeBuilder<CoefficientPrecision, CoordinatePrecision>::compute_blocks(const VirtualGenerator<CoefficientPrecision> &generator) const {

#if defined(_OPENMP) && !defined(PYTHON_INTERFACE)
#    pragma omp parallel
#endif
    {
        // std::vector<HMatrixType *> local_dense_leaves{};
        // std::vector<HMatrixType *> local_low_rank_leaves{};
        int local_false_positive = 0;
#if defined(_OPENMP) && !defined(PYTHON_INTERFACE)
#    pragma omp for schedule(guided) nowait
#endif
        for (int p = 0; p < m_admissible_tasks.size(); p++) {
            m_admissible_tasks[p]->compute_low_rank_data(generator, *m_low_rank_generator, m_reqrank, m_epsilon);
            // local_low_rank_leaves.emplace_back(m_admissible_tasks[p]);
            if (m_admissible_tasks[p]->get_low_rank_data()->get_U().nb_rows() != m_admissible_tasks[p]->get_target_cluster().get_size() || m_admissible_tasks[p]->get_low_rank_data()->get_V().nb_cols() != m_admissible_tasks[p]->get_source_cluster().get_size()) {
                // local_low_rank_leaves.pop_back();
                m_admissible_tasks[p]->clear_low_rank_data();
                // if (m_dense_blocks_generator.get() == nullptr) {
                m_admissible_tasks[p]->compute_dense_data(generator);
                // }
                // local_dense_leaves.emplace_back(m_admissible_tasks[p]);
                local_false_positive += 1;
            }
        }
        if (m_dense_blocks_generator.get() == nullptr) {
#if defined(_OPENMP) && !defined(PYTHON_INTERFACE)
#    pragma omp for schedule(guided) nowait
#endif
            for (int p = 0; p < m_dense_tasks.size(); p++) {
                m_dense_tasks[p]->compute_dense_data(generator);
            }
            // local_dense_leaves.emplace_back(m_dense_tasks[p]);
        }
#if defined(_OPENMP) && !defined(PYTHON_INTERFACE)
#    pragma omp critical
#endif
        {
            // m_low_rank_leaves.insert(m_low_rank_leaves.end(), std::make_move_iterator(local_low_rank_leaves.begin()), std::make_move_iterator(local_low_rank_leaves.end()));

            // m_dense_leaves.insert(m_dense_leaves.end(), std::make_move_iterator(local_dense_leaves.begin()), std::make_move_iterator(local_dense_leaves.end()));

            m_false_positive += local_false_positive;
        }
    }

    if (m_dense_blocks_generator.get() != nullptr) {
        std::vector<int> rows_sizes(this->m_dense_tasks.size()), cols_sizes(this->m_dense_tasks.size()), rows_offsets(this->m_dense_tasks.size()), cols_offsets(this->m_dense_tasks.size());
        std::vector<CoefficientPrecision *> ptr(this->m_dense_tasks.size());
        ZeroGenerator zero_generator;
        for (int i = 0; i < this->m_dense_tasks.size(); i++) {
            this->m_dense_tasks[i]->compute_dense_data(zero_generator);
            rows_sizes[i]   = this->m_dense_tasks[i]->get_target_cluster().get_size();
            cols_sizes[i]   = this->m_dense_tasks[i]->get_source_cluster().get_size();
            rows_offsets[i] = this->m_dense_tasks[i]->get_target_cluster().get_offset();
            cols_offsets[i] = this->m_dense_tasks[i]->get_source_cluster().get_offset();
            ptr[i]          = this->m_dense_tasks[i]->get_dense_data()->data();
        }
        if (!m_dense_tasks.empty()) {
            m_dense_blocks_generator->copy_dense_blocks(rows_sizes, cols_sizes, rows_offsets, cols_offsets, ptr);
        }
    }

    // if (m_block_diagonal_hmatrix != nullptr) {
    //     int local_offset_s = m_block_diagonal_hmatrix->get_source_cluster_tree().get_offset();
    //     int local_size_s   = m_block_diagonal_hmatrix->get_source_cluster_tree().get_size();

    //     // Build vectors of pointers for diagonal blocks
    //     for (auto leaf : m_leaves) {
    //         if (local_offset_s <= leaf->get_source_cluster_tree().get_offset() && leaf->get_source_cluster_tree().get_offset() < local_offset_s + local_size_s) {
    //             m_leaves_in_diagonal_block.push_back(leaf);
    //         }
    //     }
    //     for (auto low_rank_leaf : m_low_rank_leaves) {
    //         if (local_offset_s <= low_rank_leaf->get_source_cluster_tree().get_offset() && low_rank_leaf->get_source_cluster_tree().get_offset() < local_offset_s + local_size_s) {
    //             m_low_rank_leaves_in_diagonal_block.push_back(low_rank_leaf);
    //             if (low_rank_leaf->get_source_cluster_tree().get_offset() == low_rank_leaf->get_target_cluster_tree().get_offset()) {
    //                 m_diagonal_low_rank_leaves.push_back(low_rank_leaf);
    //             }
    //         }
    //     }
    //     for (auto dense_leaf : m_dense_leaves) {
    //         if (local_offset_s <= dense_leaf->get_source_cluster_tree().get_offset() && dense_leaf->get_source_cluster_tree().get_offset() < local_offset_s + local_size_s) {
    //             m_dense_leaves_in_diagonal_block.push_back(dense_leaf);
    //             if (dense_leaf->get_source_cluster_tree().get_offset() == dense_leaf->get_target_cluster_tree().get_offset()) {
    //                 m_diagonal_dense_leaves.push_back(dense_leaf);
    //             }
    //         }
    //     }
    // }
}

// template <typename CoefficientPrecision, typename CoordinatesPrecision, typename PreOrderFunction>
// void block_tree_preorder_traversal(const BlockTree<CoefficientPrecision, CoordinatesPrecision> &block_tree, PreOrderFunction pre_order_visitor) {
//     std::stack<const HMatrix<CoefficientPrecision, CoordinatesPrecision> *> hmatrix_stack{std::deque<const HMatrix<CoefficientPrecision, CoordinatesPrecision> *>{block_tree.get_root_hmatrix()}};

//     while (!hmatrix_stack.empty()) {
//         const HMatrix<CoefficientPrecision, CoordinatesPrecision> *current_hmatrix = hmatrix_stack.top();
//         hmatrix_stack.pop();
//         pre_order_visitor(*current_hmatrix, block_tree);

//         const auto &children = current_hmatrix->get_children();
//         for (auto child = children.rbegin(); child != children.rend(); child++) {
//             hmatrix_stack.push(*child);
//         }
//     }
// }

} // namespace htool
#endif
