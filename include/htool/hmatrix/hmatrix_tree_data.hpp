#ifndef HTOOL_HMATRIX_TREE_DATA_HPP
#define HTOOL_HMATRIX_TREE_DATA_HPP

#include "interfaces/virtual_admissibility_condition.hpp"
#include "interfaces/virtual_lrmat_generator.hpp"

namespace htool {

template <typename CoefficientPrecision, typename CoordinatePrecision>
class HMatrix;

template <typename CoefficientPrecision, typename CoordinatePrecision>
struct HMatrixTreeData {
    // Parameters
    std::shared_ptr<const Cluster<CoordinatePrecision>> m_target_cluster_tree, m_source_cluster_tree; // root clusters
    underlying_type<CoefficientPrecision> m_epsilon{1e-6};
    CoordinatePrecision m_eta{10};
    unsigned int m_maxblocksize{1000000};
    unsigned int m_minsourcedepth{0};
    unsigned int m_mintargetdepth{0};
    bool m_use_permutation{true};
    bool m_delay_dense_computation{false};
    int m_reqrank{-1};

    // Views
    const HMatrix<CoefficientPrecision, CoordinatePrecision> *m_block_diagonal_hmatrix{nullptr};
    std::vector<HMatrix<CoefficientPrecision, CoordinatePrecision> *> m_admissible_tasks{};
    std::vector<HMatrix<CoefficientPrecision, CoordinatePrecision> *> m_dense_tasks{};

    // Information
    int m_false_positive{0};

    // Strategies
    std::shared_ptr<VirtualLowRankGenerator<CoefficientPrecision, CoordinatePrecision>> m_low_rank_generator;
    std::shared_ptr<VirtualAdmissibilityCondition<CoordinatePrecision>> m_admissibility_condition;
};

} // namespace htool
#endif
