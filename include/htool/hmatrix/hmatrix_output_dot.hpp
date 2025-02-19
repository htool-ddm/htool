#ifndef HTOOL_HMATRIX_OUTPUT_DOT_HPP
#define HTOOL_HMATRIX_OUTPUT_DOT_HPP

#include "hmatrix.hpp"

/*
This file contain usefull functions for task based hmatrix factorization as described in the phD thesis of Benoit Lizé {https://theses.hal.science/tel-01244260}.
*/

namespace htool {

struct TreeCounts {
    size_t call_count = 0;
    size_t node_count = 0;
};

/**
 * @brief Visualizes the block tree of a hierarchical matrix (HMatrix).
 *
 * This function generates a DOT file representing the hierarchical structure of the given HMatrix.
 * The DOT file is then converted into an SVG image to visualize the block tree.
 * The function first determines an initial set of nodes (L0) with a maximum size of nb_nodes_max.
 * It then writes the block tree data, including nodes and edges, to a DOT file.
 * Tooltip information summarizing the block tree, such as the number of nodes and nodes in L0,
 * is added to the DOT file. Finally, the DOT file is converted to an SVG image using the
 * system's DOT command. Note that to generate a svg, you can use `dot -Tsvg block_tree.dot -o block_tree.svg`.
 *
 * @param hmatrix The input hierarchical matrix to be visualized.
 * @param nb_nodes_max The maximum number of nodes in L0.
 */
template <typename CoefficientPrecision, typename CoordinatePrecision = underlying_type<CoefficientPrecision>>
void view_block_tree(const HMatrix<CoefficientPrecision, CoordinatePrecision> &hmatrix, const std::vector<const HMatrix<CoefficientPrecision, CoordinatePrecision> *> &subhmatrices_to_color, std::ostream &dotFile) {
    // Find the initial set of nodes L0 with a maximum size of nb_nodes_max
    // std::vector<const HMatrix<CoefficientPrecision, CoordinatePrecision> *> L0 = find_l0(hmatrix, nb_nodes_max);

    // Start the DOT file content
    dotFile << "digraph {\n";

    // Create the block tree by adding nodes and edges to the DOT file
    TreeCounts counts;
    create_block_tree(hmatrix, dotFile, counts, subhmatrices_to_color);

    // Add tooltip information to the DOT file summarizing the block tree
    dotFile << "  tooltip=\"Block tree information: \\n"
            << "number of HMatrices: " << counts.node_count << "\\n"
            << "number of subHMatrices to color: " << subhmatrices_to_color.size() << "\";\n";
    dotFile << "}\n";
}

/**
 * @brief Creates a unique identifier for a given HMatrix based on its target and source cluster offsets.
 *
 * This function generates an identifier by concatenating the minimum and maximum offsets of the target and source clusters.
 * This identifier is used to label nodes in the block tree visualization of the HMatrix.
 *
 * @param hmatrix The input hierarchical matrix for which the identifier is generated.
 * @return A unique identifier for the given HMatrix.
 */
template <typename CoefficientPrecision, typename CoordinatePrecision = underlying_type<CoefficientPrecision>>
std::string get_hmatrix_id(const HMatrix<CoefficientPrecision, CoordinatePrecision> &hmatrix) {
    // Get the minimum and maximum offsets of the target and source clusters
    std::string target_min = std::to_string(hmatrix.get_target_cluster().get_offset());
    std::string target_max = std::to_string(hmatrix.get_target_cluster().get_offset() + hmatrix.get_target_cluster().get_size() - 1);
    std::string source_min = std::to_string(hmatrix.get_source_cluster().get_offset());
    std::string source_max = std::to_string(hmatrix.get_source_cluster().get_offset() + hmatrix.get_source_cluster().get_size() - 1);

    // Create the identifier by concatenating the offsets
    return target_min + "_" + target_max + "_" + source_min + "_" + source_max;
}

/**
 * @brief Adds a node to the block tree in the DOT file.
 *
 * This function adds a new node to the dot file.
 * It also adds the node information in a tooltip.
 * If the node is in subhmatrices_to_color, it colors it in light blue.
 * The function also increments the node count.
 *
 * @param hmatrix The input hierarchical matrix for which the node is created.
 * @param dotFile The output stream to write the node data to.
 * @param counts The TreeCounts object to increment the node count.
 * @param subhmatrices_to_color A vector of nodes to be highlighted.
 */
template <typename CoefficientPrecision, typename CoordinatePrecision = underlying_type<CoefficientPrecision>>
void add_node_to_block_tree(const HMatrix<CoefficientPrecision, CoordinatePrecision> &hmatrix, std::ostream &dotFile, TreeCounts &counts, const std::vector<const HMatrix<CoefficientPrecision, CoordinatePrecision> *> &subhmatrices_to_color) {
    // Define node
    dotFile << "    H_" << get_hmatrix_id(hmatrix) << " [tooltip=\"Node information: \\n";

    // Add the node information in a tooltip.
    auto hmatrix_info = get_hmatrix_information(hmatrix);
    for (const auto &info : hmatrix_info) {
        dotFile << info.first << ": " << info.second << "\\n";
    }

    // Check if the current node is in subhmatrices_to_color and color it in light blue if true
    dotFile << "\"";
    if (std::find(subhmatrices_to_color.begin(), subhmatrices_to_color.end(), &hmatrix) != subhmatrices_to_color.end()) {
        dotFile << ", style=filled, fillcolor=lightblue";
    }

    // End of node definition
    dotFile << "];\n";

    // Increment node count
    counts.node_count++;
}

/**
 * @brief Constructs a block tree visualization for a hierarchical matrix (HMatrix).
 *
 * This function generates the structure of a block tree by recursively adding nodes
 * and edges to a provided DOT file stream. It starts by adding the root node and
 * then iterates over the children of each node, adding them to the DOT file and
 * linking them with edges. The tree is constructed in a depth-first manner.
 * Each node and edge is annotated with tooltips for additional information.
 *
 * @param hmatrix The input hierarchical matrix for which the block tree is created.
 * @param dotFile The output stream to write the DOT representation of the block tree.
 * @param counts A TreeCounts object that tracks the number of calls and nodes processed.
 * @param subhmatrices_to_color A vector of pointers to HMatrix nodes, representing an initial set of nodes.
 */
template <typename CoefficientPrecision, typename CoordinatePrecision = underlying_type<CoefficientPrecision>>
void create_block_tree(const HMatrix<CoefficientPrecision, CoordinatePrecision> &hmatrix, std::ostream &dotFile, TreeCounts &counts, const std::vector<const HMatrix<CoefficientPrecision, CoordinatePrecision> *> &subhmatrices_to_color) {
    // Increment the call count
    counts.call_count++;

    // Add root node
    if (counts.call_count == 1) {
        // Add the root node to the dot file
        add_node_to_block_tree(hmatrix, dotFile, counts, subhmatrices_to_color);
    }

    // Add child nodes
    for (const auto &child : hmatrix.get_children()) {
        // Add the child node to the dot file
        add_node_to_block_tree(*child.get(), dotFile, counts, subhmatrices_to_color);

        // Add an edge between the parent and child nodes
        dotFile << "    H_" << get_hmatrix_id(hmatrix) << " -> H_" << get_hmatrix_id(*child.get()) << " [tooltip=\"" << get_hmatrix_id(hmatrix) << " -> " << get_hmatrix_id(*child.get()) << "\"];\n";

        // Recursively add child nodes
        create_block_tree(*child.get(), dotFile, counts, subhmatrices_to_color);
    }
}

} // namespace htool

#endif