#ifndef HTOOL_HMATRIX_TREE_DATA_HPP
#define HTOOL_HMATRIX_TREE_DATA_HPP

#include "../clustering/cluster_node.hpp"                 // for Cluster
#include "../misc/misc.hpp"                               // for underlying_type
#include "interfaces/virtual_admissibility_condition.hpp" // for VirtualAdmissibility ..
#include "interfaces/virtual_lrmat_generator.hpp"         // for VirtualLowRankGene...
#include <chrono>                                         // for duration
#include <map>                                            // for map
#include <memory>                                         // for shared_ptr
#include <string>                                         // for string

namespace htool {

template <typename CoefficientPrecision, typename CoordinatePrecision = underlying_type<CoefficientPrecision>>
struct HMatrixTreeData {
    // Parameters
    std::shared_ptr<const Cluster<CoordinatePrecision>> m_target_cluster_tree, m_source_cluster_tree; // root clusters
    underlying_type<CoefficientPrecision> m_epsilon{1e-6};
    CoordinatePrecision m_eta{10};
    unsigned int m_minimal_source_depth{0};
    unsigned int m_minimal_target_depth{0};
    bool m_delay_dense_computation{false};
    int m_reqrank{-1};
    bool m_is_block_tree_consistent{true};

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
