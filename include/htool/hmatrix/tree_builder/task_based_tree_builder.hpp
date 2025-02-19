#ifndef HTOOL_HMATRIX_TASK_BASED_TREE_BUILDER_HPP
#define HTOOL_HMATRIX_TASK_BASED_TREE_BUILDER_HPP

#include "../hmatrix.hpp"
namespace htool {

/**
 * @brief The cost_function associates with a node of the group tree a score
 * representing an estimate of the amount of work associated with this leaf.
 *
 * The cost of a node is given by the number of points in the target cluster
 * times the number of points in the source cluster.
 *
 * @param hmatrix The input hierarchical matrix.
 * @return The naive cost of the node, i.e. the number of points in the target cluster
 * times the number of points in the source cluster.
 */
template <typename CoefficientPrecision, typename CoordinatePrecision = underlying_type<CoefficientPrecision>>
std::size_t cost_function(const HMatrix<CoefficientPrecision, CoordinatePrecision> &hmatrix) {
    std::size_t nb_rows = hmatrix.get_target_cluster().get_size();
    std::size_t nb_cols = hmatrix.get_source_cluster().get_size();
    return std::size_t(nb_rows * nb_cols);
}

// overload for const Cluster
template <typename CoordinatePrecision>
std::size_t cost_function(const Cluster<CoordinatePrecision> &cluster) {
    return std::size_t(cluster.get_size());
}

/**
 * @brief Find the nodes of the group tree such that the total cost of their
 * leaves is less than or equal to nb_nodes_max.
 *
 * This function performs a dichotomy search to find the optimal criterion
 * such that the total cost of the nodes of the group tree is less than or
 * equal to nb_nodes_max.
 *
 * @param root_hmatrix The root node of the group tree.
 * @param nb_nodes_max The maximum total cost of the nodes of the group tree.
 * @return A vector of pointers to the nodes of the group tree such that the
 * total cost of their leaves is less than or equal to nb_nodes_max.
 */
template <typename CoefficientPrecision, typename CoordinatePrecision = underlying_type<CoefficientPrecision>>
std::vector<HMatrix<CoefficientPrecision, CoordinatePrecision> *> find_l0(HMatrix<CoefficientPrecision, CoordinatePrecision> &root_hmatrix, const size_t nb_nodes_max) {
    // Initialize criterion with the cost of the root node
    double criterion = cost_function(root_hmatrix);

    // Find initial nodes that meet the criterion
    std::vector<HMatrix<CoefficientPrecision, CoordinatePrecision> *> old_result, result = count_nodes(root_hmatrix, criterion);
    // Check if the initial result exceeds the maximum allowed nodes
    if (result.size() > nb_nodes_max) {
        std::cerr << "Error: no L0 can be defined." << std::endl;
        return std::vector<HMatrix<CoefficientPrecision, CoordinatePrecision> *>();
    } else {
        // Perform a dichotomy search to find the optimal criterion
        do {
            // Save the result of this iteration
            old_result = result;

            // If all nodes are leaves, return the result as it cannot be further divided
            if (std::all_of(result.begin(), result.end(), [](HMatrix<CoefficientPrecision, CoordinatePrecision> *hmatrix) {
                    return hmatrix->is_leaf();
                })) {
                return result;
            }

            // Reduce the criterion to explore smaller cost nodes
            criterion /= 2;

            // Update the result with the new criterion
            result = count_nodes(root_hmatrix, criterion);
        } while (result.size() <= nb_nodes_max); // Loop until the result size exceeds nb_nodes_max
    }

    // Return the last valid result
    return old_result;
}

// overload for const Cluster
template <typename CoordinatePrecision>
std::vector<const Cluster<CoordinatePrecision> *> find_l0(const Cluster<CoordinatePrecision> &cluster, const size_t nb_nodes_max) {
    // Initialize criterion with the cost of the root node
    double criterion = cost_function(cluster);

    // Find initial nodes that meet the criterion
    std::vector<const Cluster<CoordinatePrecision> *> old_result, result = count_nodes(cluster, criterion);

    // Check if the initial result exceeds the maximum allowed nodes
    if (result.size() > nb_nodes_max) {
        std::cerr << "Error: no L0 can be defined." << std::endl;
        return std::vector<const Cluster<CoordinatePrecision> *>();
    } else {
        // Perform a dichotomy search to find the optimal criterion
        do {
            // Save the result of this iteration
            old_result = result;

            // If all nodes are leaves, return the result as it cannot be further divided
            if (std::all_of(result.begin(), result.end(), [](const Cluster<CoordinatePrecision> *leaf_cluster) {
                    return leaf_cluster->is_leaf();
                })) {
                return result;
            }

            // Reduce the criterion to explore smaller cost nodes
            criterion /= 2;

            // Update the result with the new criterion
            result = count_nodes(cluster, criterion);
        } while (result.size() <= nb_nodes_max); // Loop until the result size exceeds nb_nodes_max
    }

    // Return the last valid result
    return old_result;
}

/**
 * @brief Performs a postorder tree traversal of the group tree and returns a vector of all the nodes of the group tree that have a cost less than criterion.
 *
 * The cost of a node is given by the cost_function: it is the product of the number of points in the target cluster and the number of points in the source cluster.
 *
 * The criterion is the maximum cost of the nodes to be returned.
 *
 * The result is a vector of pointers to the nodes of the group tree that have a cost less than criterion.
 *
 * @param hmatrix The input hierarchical matrix.
 * @param criterion The maximum cost of the nodes to be returned.
 * @return A vector of pointers to the nodes of the group tree that have a cost less than criterion.
 */
template <typename CoefficientPrecision, typename CoordinatePrecision = underlying_type<CoefficientPrecision>>
std::vector<HMatrix<CoefficientPrecision, CoordinatePrecision> *> count_nodes(HMatrix<CoefficientPrecision, CoordinatePrecision> &hmatrix, double criterion) {
    std::vector<HMatrix<CoefficientPrecision, CoordinatePrecision> *> result;

    if (cost_function(hmatrix) <= criterion || hmatrix.is_leaf()) {
        // if the node is a leaf or its cost is less than criterion, add it to the result
        result.push_back(&hmatrix);
    } else {
        // if the node is not a leaf, traverse its children
        for (auto &child : hmatrix.get_children()) {
            // perform a postorder tree traversal of the subtree rooted at child
            auto local_result = count_nodes(*child.get(), criterion);
            // add the result of the subtree traversal to the result
            result.insert(result.end(), local_result.begin(), local_result.end());
        }
    }
    return result;
}

// overload for const Cluster
template <typename CoordinatePrecision>
std::vector<const Cluster<CoordinatePrecision> *> count_nodes(const Cluster<CoordinatePrecision> &cluster, double criterion) {
    std::vector<const Cluster<CoordinatePrecision> *> result;

    if (cost_function(cluster) <= criterion || cluster.is_leaf()) {
        // if the node is a leaf or its cost is less than criterion, add it to the result
        result.push_back(&cluster);
    } else {
        // if the node is not a leaf, traverse its children
        for (const auto &child : cluster.get_children()) {
            // perform a postorder tree traversal of the subtree rooted at child
            auto local_result = count_nodes(*child.get(), criterion);
            // add the result of the subtree traversal to the result
            result.insert(result.end(), local_result.begin(), local_result.end());
        }
    }
    return result;
}

/**
 * Enumerates the dependences of a given hierarchical matrix (HMatrix) with respect to a set of nodes (L0).
 *
 * This function determines the relationship of the input HMatrix with the nodes in the L0 vector,
 * based on their hierarchical position, and returns a vector of dependent HMatrix pointers.
 *
 * Cases handled:
 * 1. If the input HMatrix is present in L0, it directly returns the HMatrix.
 * 2. If the input HMatrix is an ancestor of any node in L0, it returns those descendant nodes from L0.
 * 3. If the input HMatrix is a descendant of any node in L0, it returns the unique ancestor from L0.
 * 4. If none of the above cases are satisfied, it logs an error indicating an unexpected state.
 *
 * @param hmatrix The hierarchical matrix for which dependencies are to be enumerated.
 * @param L0 A vector of hierarchical matrix pointers representing a set of nodes.
 * @return A vector of pointers to the hierarchical matrices that are dependencies of the input matrix.
 */

template <typename CoefficientPrecision, typename CoordinatePrecision = underlying_type<CoefficientPrecision>>
std::vector<const HMatrix<CoefficientPrecision, CoordinatePrecision> *> enumerate_dependences(const HMatrix<CoefficientPrecision, CoordinatePrecision> &hmatrix, const std::vector<HMatrix<CoefficientPrecision, CoordinatePrecision> *> &L0) {
    // Case 1 : hmatrix is in L0
    if (std::find(L0.begin(), L0.end(), &hmatrix) != L0.end()) {
        return {&hmatrix};
    }

    // Case 2 : hmatrix is above L0. Find descendants of hmatrix in L0
    std::vector<const HMatrix<CoefficientPrecision, CoordinatePrecision> *> children;
    for (const auto &hmatrix_on_L0 : L0) {
        if (left_hmatrix_ancestor_of_right_hmatrix(hmatrix, *hmatrix_on_L0)) {
            children.push_back(hmatrix_on_L0);
        }
    }
    if (!children.empty()) {
        return children;
    }

    // Case 3 : hmatrix is below L0. Find the unique ancestor of hmatrix in L0

    const auto it = std::find_if(L0.begin(), L0.end(), [&hmatrix](HMatrix<CoefficientPrecision, CoordinatePrecision> *hmatrix_in_L0) {
        return left_hmatrix_descendant_of_right_hmatrix(hmatrix, *hmatrix_in_L0);
    });

    if (it != L0.end()) {
        return {*it};
    }

    // Case 4 : error
    Logger::get_instance().log(LogLevel::ERROR, "No dependence found with L0. It should not happen.");
    return {};
}

// overload for non-const HMatrix pointers
template <typename CoefficientPrecision, typename CoordinatePrecision = underlying_type<CoefficientPrecision>>
std::vector<HMatrix<CoefficientPrecision, CoordinatePrecision> *> enumerate_dependences(HMatrix<CoefficientPrecision, CoordinatePrecision> &hmatrix, const std::vector<HMatrix<CoefficientPrecision, CoordinatePrecision> *> &L0) {
    // Case 1 : hmatrix is in L0
    if (std::find(L0.begin(), L0.end(), &hmatrix) != L0.end()) {
        return {&hmatrix};
    }

    // Case 2 : hmatrix is above L0. Find descendants of hmatrix in L0
    std::vector<HMatrix<CoefficientPrecision, CoordinatePrecision> *> children;
    for (auto &hmatrix_on_L0 : L0) {
        if (left_hmatrix_ancestor_of_right_hmatrix(hmatrix, *hmatrix_on_L0)) {
            children.push_back(hmatrix_on_L0);
        }
    }
    if (!children.empty()) {
        return children;
    }

    // Case 3 : hmatrix is below L0. Find the unique ancestor of hmatrix in L0

    const auto it = std::find_if(L0.begin(), L0.end(), [&hmatrix](HMatrix<CoefficientPrecision, CoordinatePrecision> *hmatrix_in_L0) {
        return left_hmatrix_descendant_of_right_hmatrix(hmatrix, *hmatrix_in_L0);
    });

    if (it != L0.end()) {
        return {*it};
    }

    // Case 4 : error
    Logger::get_instance().log(LogLevel::ERROR, "No dependence found with L0. It should not happen.");
    return {};
}

// overload for const Cluster pointers
template <typename CoordinatePrecision>
std::vector<const Cluster<CoordinatePrecision> *> enumerate_dependences(const Cluster<CoordinatePrecision> &cluster, const std::vector<const Cluster<CoordinatePrecision> *> &L0) {
    // Case 1 : cluster is in L0
    if (std::find(L0.begin(), L0.end(), &cluster) != L0.end()) {
        return {&cluster};
    }

    // Case 2 : cluster is above L0. Find descendants of cluster in L0
    std::vector<const Cluster<CoordinatePrecision> *> children;
    for (const auto &cluster_on_L0 : L0) {
        if (left_cluster_contains_right_cluster(cluster, *cluster_on_L0)) {
            children.push_back(cluster_on_L0);
        }
    }
    if (!children.empty()) {
        return children;
    }

    // Case 3 : cluster is below L0. Find the unique ancestor of cluster in L0
    const auto it = std::find_if(L0.begin(), L0.end(), [&cluster](const Cluster<CoordinatePrecision> *cluster_in_L0) {
        return left_cluster_contains_right_cluster(*cluster_in_L0, cluster);
    });
    if (it != L0.end()) {
        return {*it};
    }

    // Case 4 : error
    Logger::get_instance().log(LogLevel::ERROR, "No dependence found with L0. It should not happen.");
    return {};
}

} // namespace htool

#endif