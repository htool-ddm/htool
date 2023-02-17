#ifndef HTOOL_HMATRIX_BLOCK_TREE_OUTPUT_HPP
#define HTOOL_HMATRIX_BLOCK_TREE_OUTPUT_HPP

#include <iostream>

namespace htool {

template <typename CoefficientPrecision, typename CoordinatesPrecision>
class HMatrix;

template <typename OutputValue>
struct DisplayBlock {
    int target_offset;
    int source_offset;
    int target_size;
    int source_size;
    OutputValue output_value;
};

template <typename OutputValue>
std::ostream &operator<<(std::ostream &output_stream, const DisplayBlock<OutputValue> &display_block) {
    output_stream << display_block.target_offset << "," << display_block.target_size << "," << display_block.source_offset << "," << display_block.source_size << "," << display_block.output_value;
    return output_stream;
}

template <typename CoefficientPrecision, typename CoordinatesPrecision>
void save_leaves_with_rank(const HMatrix<CoefficientPrecision, CoordinatesPrecision> &hmatrix, std::string filename) {
    std::ofstream output(filename + ".csv");
    std::vector<DisplayBlock<int>> output_blocks{};

    preorder_tree_traversal(
        hmatrix,
        [&output_blocks, &hmatrix](const HMatrix<CoefficientPrecision, CoordinatesPrecision> &current_hmatrix) {
            if (current_hmatrix.is_leaf()) {
                output_blocks.push_back(DisplayBlock<int>{current_hmatrix.get_target_cluster().get_offset() - hmatrix.get_target_cluster().get_offset(), current_hmatrix.get_source_cluster().get_offset() - hmatrix.get_source_cluster().get_offset(), current_hmatrix.get_target_cluster().get_size(), current_hmatrix.get_source_cluster().get_size(), current_hmatrix.get_rank()});
            }
        });
    output << hmatrix.get_target_cluster().get_size() << ",";
    output << hmatrix.get_source_cluster().get_size() << "\n";
    for (const auto &block : output_blocks) {
        output << block << "\n";
    }
}

template <typename CoefficientPrecision, typename CoordinatesPrecision>
void save_levels(const HMatrix<CoefficientPrecision, CoordinatesPrecision> &hmatrix, std::string filename, std::vector<int> depths) {
    std::vector<std::vector<DisplayBlock<int>>> output_blocks(depths.size());

    preorder_tree_traversal(
        hmatrix,
        [&output_blocks, &depths, &hmatrix](const HMatrix<CoefficientPrecision, CoordinatesPrecision> &current_hmatrix) {
            auto it = std::find(depths.begin(), depths.end(), current_hmatrix.get_depth());

            if (it != depths.end()) {
                int index = std::distance(depths.begin(), it);
                output_blocks[index].push_back(DisplayBlock<int>{current_hmatrix.get_target_cluster().get_offset() - hmatrix.get_target_cluster().get_offset(), current_hmatrix.get_source_cluster().get_offset() - hmatrix.get_source_cluster().get_offset(), current_hmatrix.get_target_cluster().get_size(), current_hmatrix.get_source_cluster().get_size(), current_hmatrix.get_rank()});
            }
        });

    for (int p = 0; p < depths.size(); p++) {
        std::ofstream output(filename + std::to_string(p) + ".csv");
        output << hmatrix.get_target_cluster().get_size() << ",";
        output << hmatrix.get_source_cluster().get_size() << "\n";

        for (const auto &block : output_blocks[p]) {
            output << block << "\n";
        }
        output << "\n";
    }
}

} // namespace htool

#endif
