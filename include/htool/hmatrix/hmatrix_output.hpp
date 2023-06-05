#ifndef HTOOL_HMATRIX_OUTPUT_HPP
#define HTOOL_HMATRIX_OUTPUT_HPP

#include <array>
#include <iostream>
#if defined(_OPENMP)
#    include <omp.h>
#endif

namespace htool {

template <typename CoefficientPrecision, typename CoordinatePrecision>
class HMatrix;

template <typename CoefficientPrecision, typename CoordinatePrecision>
struct HMatrixTreeData;

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

template <typename CoefficientPrecision, typename CoordinatePrecision>
void save_leaves_with_rank(const HMatrix<CoefficientPrecision, CoordinatePrecision> &hmatrix, std::string filename) {
    std::ofstream output(filename + ".csv");
    std::vector<DisplayBlock<int>> output_blocks{};

    preorder_tree_traversal(
        hmatrix,
        [&output_blocks, &hmatrix](const HMatrix<CoefficientPrecision, CoordinatePrecision> &current_hmatrix) {
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

template <typename CoefficientPrecision, typename CoordinatePrecision>
void save_levels(const HMatrix<CoefficientPrecision, CoordinatePrecision> &hmatrix, std::string filename, std::vector<int> depths) {
    std::vector<std::vector<DisplayBlock<int>>> output_blocks(depths.size());

    preorder_tree_traversal(
        hmatrix,
        [&output_blocks, &depths, &hmatrix](const HMatrix<CoefficientPrecision, CoordinatePrecision> &current_hmatrix) {
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

template <typename CoefficientPrecision, typename CoordinatePrecision>
void print_tree_parameters(const HMatrix<CoefficientPrecision, CoordinatePrecision> &hmatrix, std::ostream &os) {
    const HMatrixTreeData<CoefficientPrecision, CoordinatePrecision> *hmatrix_tree_data = hmatrix.get_hmatrix_tree_data();

    int output_size = 23;
    os << "Block tree parameters\n";
    os << std::setw(output_size) << std::setfill('_') << "Eta" << hmatrix_tree_data->m_eta << "\n";
    os << std::setw(output_size) << std::setfill('_') << "Epsilon" << hmatrix_tree_data->m_epsilon << "\n";
    os << std::setw(output_size) << std::setfill('_') << "MaxBlockSize" << hmatrix_tree_data->m_maxblocksize << "\n";
    os << std::setw(output_size) << std::setfill('_') << "MinTargetDepth" << hmatrix_tree_data->m_minimal_target_depth << "\n";
    os << std::setw(output_size) << std::setfill('_') << "MinClusterSizeTarget" << hmatrix.get_target_cluster().get_minclustersize() << "\n";
    os << std::setw(output_size) << std::setfill('_') << "MaxClusterDepthTarget" << hmatrix.get_target_cluster().get_maximal_depth() << "\n";
    os << std::setw(output_size) << std::setfill('_') << "MinClusterDepthTarget" << hmatrix.get_target_cluster().get_minimal_depth() << "\n";
    os << std::setw(output_size) << std::setfill('_') << "MinSourceDepth" << hmatrix_tree_data->m_minimal_source_depth << "\n";
    os << std::setw(output_size) << std::setfill('_') << "MinClusterSizeSource" << hmatrix.get_source_cluster().get_minclustersize() << "\n";
    os << std::setw(output_size) << std::setfill('_') << "MaxClusterDepthSource" << hmatrix.get_source_cluster().get_maximal_depth() << "\n";
    os << std::setw(output_size) << std::setfill('_') << "MinClusterDepthSource" << hmatrix.get_source_cluster().get_minimal_depth() << "\n";
    os << "\n";
}

template <typename CoefficientPrecision, typename CoordinatePrecision>
void get_leaves(const HMatrix<CoefficientPrecision, CoordinatePrecision> &hmatrix, std::vector<const HMatrix<CoefficientPrecision, CoordinatePrecision> *> &dense_blocks, std::vector<const HMatrix<CoefficientPrecision, CoordinatePrecision> *> &low_rank_blocks) {
    preorder_tree_traversal(
        hmatrix,
        [&dense_blocks, &low_rank_blocks](const HMatrix<CoefficientPrecision, CoordinatePrecision> &current_hmatrix) {
            if (current_hmatrix.is_leaf() && current_hmatrix.is_dense()) {
                dense_blocks.push_back(&current_hmatrix);
            } else if (current_hmatrix.is_leaf() && current_hmatrix.is_low_rank()) {
                low_rank_blocks.push_back(&current_hmatrix);
            }
        });
}

template <typename CoefficientPrecision, typename CoordinatePrecision>
void print_hmatrix_information(const HMatrix<CoefficientPrecision, CoordinatePrecision> &hmatrix, std::ostream &os) {
    const HMatrixTreeData<CoefficientPrecision, CoordinatePrecision> *hmatrix_tree_data = hmatrix.get_hmatrix_tree_data();

    unsigned int nb_rows = hmatrix.get_target_cluster().get_size();
    unsigned int nb_cols = hmatrix.get_source_cluster().get_size();

    // 0 : dense mat ; 1 : lr mat ; 2 : rank ; 3 : local_size
    std::array<std::size_t, 3> maxinfos = {0, 0, 0};
    std::array<double, 3> meaninfos     = {0, 0, 0};
    std::array<std::size_t, 3> mininfos = {std::max(nb_cols, nb_rows), std::max(nb_cols, nb_rows), std::max(nb_cols, nb_rows)};
    std::vector<const HMatrix<CoefficientPrecision, CoordinatePrecision> *> dense_blocks;
    std::vector<const HMatrix<CoefficientPrecision, CoordinatePrecision> *> low_rank_blocks;
    get_leaves(hmatrix, dense_blocks, low_rank_blocks);

    // Compute information
    double local_number_of_rows, local_number_of_cols;
    std::size_t local_size, local_rank;
    double number_of_generated_coefficient = 0;
    for (const auto &low_rank_block : low_rank_blocks) {
        local_number_of_rows = low_rank_block->get_target_cluster().get_size();
        local_number_of_cols = low_rank_block->get_source_cluster().get_size();
        local_size           = local_number_of_rows * local_number_of_cols;
        local_rank           = low_rank_block->get_low_rank_data()->rank_of();
        maxinfos[1]          = std::max(maxinfos[1], local_size);
        mininfos[1]          = std::min(mininfos[1], local_size);
        meaninfos[1] += local_size;
        maxinfos[2] = std::max(maxinfos[2], local_rank);
        mininfos[2] = std::min(mininfos[2], local_rank);
        meaninfos[2] += local_rank;
        number_of_generated_coefficient += local_rank * (local_number_of_rows + local_number_of_cols);
    }
    for (const auto &dense_block : dense_blocks) {
        local_number_of_rows = dense_block->get_target_cluster().get_size();
        local_number_of_cols = dense_block->get_source_cluster().get_size();
        local_size           = local_number_of_rows * local_number_of_cols;
        maxinfos[0]          = std::max(maxinfos[0], local_size);
        mininfos[0]          = std::min(mininfos[0], local_size);
        meaninfos[0] += local_size;
        if (dense_block->get_symmetry() != 'N') {
            number_of_generated_coefficient += (local_number_of_rows * (local_number_of_cols + 1)) / 2.;
        } else {
            number_of_generated_coefficient += (local_number_of_rows * local_number_of_cols);
        }
    }

    meaninfos[0] = (dense_blocks.size() == 0 ? 0 : meaninfos[0] / dense_blocks.size());
    meaninfos[1] = (low_rank_blocks.size() == 0 ? 0 : meaninfos[1] / low_rank_blocks.size());
    meaninfos[2] = (low_rank_blocks.size() == 0 ? 0 : meaninfos[2] / low_rank_blocks.size());
    mininfos[0]  = (dense_blocks.size() == 0 ? 0 : mininfos[0]);
    mininfos[1]  = (low_rank_blocks.size() == 0 ? 0 : mininfos[1]);
    mininfos[2]  = (low_rank_blocks.size() == 0 ? 0 : mininfos[2]);

    // Print parameters
    std::size_t output_size = 25;
    output_size             = std::max(output_size, 2 + std::max_element(std::begin(hmatrix_tree_data->m_information), std::end(hmatrix_tree_data->m_information), [](const auto &a, const auto &b) { return a.first.size() < b.first.size(); })->first.size());
    output_size             = std::max(output_size, 2 + std::max_element(std::begin(hmatrix_tree_data->m_timings), std::end(hmatrix_tree_data->m_timings), [](const auto &a, const auto &b) { return a.first.size() < b.first.size(); })->first.size());

    // Print
    std::ostringstream output;
    output << std::setfill('_') << std::left;
    output << "Hmatrix information\n";
    output << std::setw(output_size) << "Target_size" << nb_rows << "\n";
    output << std::setw(output_size) << "Source_size" << nb_cols << "\n";
    output << std::setw(output_size) << "Dense_block_size_max" << maxinfos[0] << "\n";
    output << std::setw(output_size) << "Dense_block_size_mean" << meaninfos[0] << "\n";
    output << std::setw(output_size) << "Dense_block_size_min" << mininfos[0] << "\n";
    output << std::setw(output_size) << "Low_rank_block_size_max" << maxinfos[1] << "\n";
    output << std::setw(output_size) << "Low_rank_block_size_mean" << meaninfos[1] << "\n";
    output << std::setw(output_size) << "Low_rank_block_size_min" << mininfos[1] << "\n";
    output << std::setw(output_size) << "Rank_max" << maxinfos[2] << "\n";
    output << std::setw(output_size) << "Rank_mean" << meaninfos[2] << "\n";
    output << std::setw(output_size) << "Rank_min" << mininfos[2] << "\n";
    output << std::setw(output_size) << "Number_of_low_rank_blocks" << low_rank_blocks.size() << "\n";
    output << std::setw(output_size) << "Number_of_dense_blocks" << dense_blocks.size() << "\n";
    output << std::setw(output_size) << "Compression_ratio" << (nb_rows * nb_cols) / number_of_generated_coefficient << "\n";
    output << std::setw(output_size) << "Space_saving" << 1 - number_of_generated_coefficient / (nb_rows * nb_cols) << "\n";
#if defined(_OPENMP)
    output << std::setw(output_size) << "Number_of_threads" << omp_get_max_threads() << "\n";
#endif
    for (const auto &elt : hmatrix_tree_data->m_information) {
        output << std::setw(output_size) << elt.first << elt.second << "\n";
    }
    for (const auto &elt : hmatrix_tree_data->m_timings) {
        output << std::setw(output_size) << elt.first << elt.second.count() << " second(s)\n";
    }
    os << output.str();
}

} // namespace htool

#endif
