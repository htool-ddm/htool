#ifndef HTOOL_HMATRIX_TREE_DATA_HPP
#define HTOOL_HMATRIX_TREE_DATA_HPP

#include "interfaces/virtual_admissibility_condition.hpp"
#include "interfaces/virtual_lrmat_generator.hpp"
#include <chrono>
#include <map>

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
    unsigned int m_minimal_source_depth{0};
    unsigned int m_minimal_target_depth{0};
    bool m_delay_dense_computation{false};
    int m_reqrank{-1};

    // Information
    mutable std::map<std::string, std::string> m_information;
    mutable std::map<std::string, std::chrono::duration<double>> m_timings;

    // Strategies
    std::shared_ptr<VirtualLowRankGenerator<CoefficientPrecision, CoordinatePrecision>>
        m_low_rank_generator;
    std::shared_ptr<VirtualAdmissibilityCondition<CoordinatePrecision>> m_admissibility_condition;
};

} // namespace htool
#endif
