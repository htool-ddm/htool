#ifndef HTOOL_HMATRIX_DISTRIBUTED_OUTPUT_HPP
#define HTOOL_HMATRIX_DISTRIBUTED_OUTPUT_HPP

#include "../wrappers/wrapper_mpi.hpp"
#include "hmatrix_output.hpp"
#include <array>
#include <iostream>
#include <mpi.h>

#if defined(_OPENMP)
#    include <omp.h>
#endif

namespace htool {

template <typename CoefficientPrecision, typename CoordinatePrecision>
class HMatrix;

template <typename CoefficientPrecision, typename CoordinatePrecision>
struct HMatrixTreeData;

bool is_positive_integer(const std::string &s) {
    return !s.empty() && std::find_if(s.begin(), s.end(), [](unsigned char c) { return !std::isdigit(c); }) == s.end();
}

template <typename CoefficientPrecision, typename CoordinatePrecision>
void print_distributed_hmatrix_information(const HMatrix<CoefficientPrecision, CoordinatePrecision> &hmatrix, std::ostream &os, MPI_Comm comm) {
    int sizeWorld, rankWorld;
    MPI_Comm_rank(comm, &rankWorld);
    MPI_Comm_size(comm, &sizeWorld);

    const HMatrixTreeData<CoefficientPrecision, CoordinatePrecision> *hmatrix_tree_data = hmatrix.get_hmatrix_tree_data();

    unsigned int local_nb_rows = hmatrix.get_target_cluster().get_size();
    unsigned int local_nb_cols = hmatrix.get_source_cluster().get_size();
    std::size_t local_size     = local_nb_cols * local_nb_rows;

    // 0 : dense mat size ; 1 : lr mat size ; 2 : rank ; 3 : nb_rows, 4: nb_cols
    std::array<std::size_t, 5> maxinfos = {0, 0, 0, local_nb_rows, local_nb_cols};
    std::array<double, 5> meaninfos     = {0, 0, 0, double(local_nb_rows), double(local_nb_cols)};
    std::array<std::size_t, 5> mininfos = {local_size, local_size, local_size, local_nb_rows, local_nb_cols};

    std::vector<const HMatrix<CoefficientPrecision, CoordinatePrecision> *> local_dense_blocks;
    std::vector<const HMatrix<CoefficientPrecision, CoordinatePrecision> *> local_low_rank_blocks;
    get_leaves(hmatrix, local_dense_blocks, local_low_rank_blocks);

    // Compute information
    std::size_t block_number_of_rows, block_number_of_cols;
    std::size_t block_size, block_rank, local_number_of_generated_coefficients{0};
    for (const auto &low_rank_block : local_low_rank_blocks) {
        block_number_of_rows = low_rank_block->get_target_cluster().get_size();
        block_number_of_cols = low_rank_block->get_source_cluster().get_size();
        block_size           = block_number_of_rows * block_number_of_cols;
        block_rank           = low_rank_block->get_low_rank_data()->rank_of();
        maxinfos[1]          = std::max(maxinfos[1], block_size);
        mininfos[1]          = std::min(mininfos[1], block_size);
        meaninfos[1] += block_size;
        maxinfos[2] = std::max(maxinfos[2], block_rank);
        mininfos[2] = std::min(mininfos[2], block_rank);
        meaninfos[2] += block_rank;
        local_number_of_generated_coefficients += block_rank * (block_number_of_rows + block_number_of_cols);
    }
    for (const auto &dense_block : local_dense_blocks) {
        block_number_of_rows = dense_block->get_target_cluster().get_size();
        block_number_of_cols = dense_block->get_source_cluster().get_size();
        block_size           = block_number_of_rows * block_number_of_cols;
        maxinfos[0]          = std::max(maxinfos[0], block_size);
        mininfos[0]          = std::min(mininfos[0], block_size);
        meaninfos[0] += block_size;
        if (dense_block->get_symmetry() != 'N') {
            local_number_of_generated_coefficients += (block_number_of_rows * (block_number_of_cols + 1)) / 2.;
        } else {
            local_number_of_generated_coefficients += (block_number_of_rows * block_number_of_cols);
        }
    }

    // 0: nb dense mat, 1 : nb lrmat, 2 total size, 3 nb generated coefficient
    std::array<std::size_t, 4> reduced_information = {
        local_dense_blocks.size(),
        local_low_rank_blocks.size(),
        local_size,
        local_number_of_generated_coefficients};

    if (rankWorld == 0) {
        MPI_Reduce(MPI_IN_PLACE, maxinfos.data(), maxinfos.size(), my_MPI_SIZE_T, MPI_MAX, 0, comm);
        MPI_Reduce(MPI_IN_PLACE, mininfos.data(), mininfos.size(), my_MPI_SIZE_T, MPI_MIN, 0, comm);
        MPI_Reduce(MPI_IN_PLACE, meaninfos.data(), meaninfos.size(), MPI_DOUBLE, MPI_SUM, 0, comm);
        MPI_Reduce(MPI_IN_PLACE, reduced_information.data(), reduced_information.size(), my_MPI_SIZE_T, MPI_SUM, 0, comm);
        // MPI_Reduce(MPI_IN_PLACE, &(false_positive), 1, MPI_INT, MPI_SUM, 0, comm);
    } else {
        MPI_Reduce(maxinfos.data(), maxinfos.data(), maxinfos.size(), my_MPI_SIZE_T, MPI_MAX, 0, comm);
        MPI_Reduce(mininfos.data(), mininfos.data(), mininfos.size(), my_MPI_SIZE_T, MPI_MIN, 0, comm);
        MPI_Reduce(meaninfos.data(), meaninfos.data(), meaninfos.size(), MPI_DOUBLE, MPI_SUM, 0, comm);
        MPI_Reduce(reduced_information.data(), reduced_information.data(), reduced_information.size(), my_MPI_SIZE_T, MPI_SUM, 0, comm);
        // MPI_Reduce(&(false_positive), &(false_positive), 1, MPI_INT, MPI_SUM, 0, comm);
    }

    std::size_t nb_dense_blocks                        = reduced_information[0];
    std::size_t nb_low_rank_blocks                     = reduced_information[1];
    std::size_t total_size                             = reduced_information[2];
    std::size_t total_number_of_generated_coefficients = reduced_information[3];

    meaninfos[0] = (nb_dense_blocks == 0 ? 0 : meaninfos[0] / nb_dense_blocks);
    meaninfos[1] = (nb_low_rank_blocks == 0 ? 0 : meaninfos[1] / nb_low_rank_blocks);
    meaninfos[2] = (nb_low_rank_blocks == 0 ? 0 : meaninfos[2] / nb_low_rank_blocks);
    meaninfos[3] = meaninfos[3] / sizeWorld;
    meaninfos[4] = meaninfos[4] / sizeWorld;

    // Print parameters
    std::size_t output_size = 25;
    output_size             = std::max(output_size, 7 + std::max_element(std::begin(hmatrix_tree_data->m_information), std::end(hmatrix_tree_data->m_information), [](const auto &a, const auto &b) { return a.first.size() < b.first.size(); })->first.size());
    output_size             = std::max(output_size, 7 + std::max_element(std::begin(hmatrix_tree_data->m_timings), std::end(hmatrix_tree_data->m_timings), [](const auto &a, const auto &b) { return a.first.size() < b.first.size(); })->first.size());

    // Print
    std::ostringstream output;
    output << std::setfill('_') << std::left;
    if (rankWorld == 0) {
        output << "Distributed Hmatrix information\n";
        output << std::setw(output_size) << "Target_size_max" << maxinfos[3] << "\n";
        output << std::setw(output_size) << "Target_size_mean" << meaninfos[3] << "\n";
        output << std::setw(output_size) << "Target_size_min" << mininfos[3] << "\n";
        output << std::setw(output_size) << "Source_size_max" << maxinfos[4] << "\n";
        output << std::setw(output_size) << "Source_size_mean" << meaninfos[4] << "\n";
        output << std::setw(output_size) << "Source_size_min" << mininfos[4] << "\n";
        output << std::setw(output_size) << "Dense_block_size_max" << maxinfos[0] << "\n";
        output << std::setw(output_size) << "Dense_block_size_mean" << meaninfos[0] << "\n";
        output << std::setw(output_size) << "Dense_block_size_min" << mininfos[0] << "\n";
        output << std::setw(output_size) << "Low_rank_block_size_max" << maxinfos[1] << "\n";
        output << std::setw(output_size) << "Low_rank_block_size_mean" << meaninfos[1] << "\n";
        output << std::setw(output_size) << "Low_rank_block_size_min" << mininfos[1] << "\n";
        output << std::setw(output_size) << "Rank_max" << maxinfos[2] << "\n";
        output << std::setw(output_size) << "Rank_mean" << meaninfos[2] << "\n";
        output << std::setw(output_size) << "Rank_min" << mininfos[2] << "\n";
        output << std::setw(output_size) << "Number_of_low_rank_blocks" << nb_low_rank_blocks << "\n";
        output << std::setw(output_size) << "Number_of_dense_blocks" << nb_dense_blocks << "\n";
        output << std::setw(output_size) << "Compression_ratio" << (total_size) / static_cast<double>(total_number_of_generated_coefficients) << "\n";
        output << std::setw(output_size) << "Space_saving" << 1 - static_cast<double>(total_number_of_generated_coefficients) / (total_size) << "\n";
        output << std::setw(output_size) << "Number_of_MPI_tasks" << sizeWorld << "\n";
#if defined(_OPENMP)
        output << std::setw(output_size) << "Number_of_threads_per_tasks" << omp_get_max_threads() << "\n";
        output << std::setw(output_size) << "Number_of_procs" << sizeWorld * omp_get_max_threads() << "\n";
#else
        output << std::setw(output_size) << "Number_of_procs" << sizeWorld << "\n";
#endif
    }

    // Reduce information
    std::vector<std::string> information_name;
    std::vector<int> information_value;
    for (const auto &elt : hmatrix_tree_data->m_information) {
        if (is_positive_integer(elt.second)) {
            information_name.push_back(elt.first);
            information_value.push_back(std::stoi(elt.second));
        }
    }

    if (rankWorld == 0) {
        MPI_Reduce(MPI_IN_PLACE, information_value.data(), information_value.size(), MPI_INT, MPI_SUM, 0, comm);
    } else {
        MPI_Reduce(information_value.data(), information_value.data(), information_value.size(), MPI_INT, MPI_SUM, 0, comm);
    }

    if (rankWorld == 0) {
        for (std::size_t i = 0; i < information_name.size(); i++) {
            output << std::setw(output_size) << information_name[i] << information_value[i] << "\n";
        }
    }

    // Reduce results about timing
    std::size_t number_of_timings = hmatrix_tree_data->m_timings.size();
    std::vector<double> max_timings(number_of_timings), mean_timings(number_of_timings), min_timings(number_of_timings);
    std::vector<std::string> timing_names(number_of_timings);
    int counter = 0;
    for (const auto &elt : hmatrix_tree_data->m_timings) {
        timing_names[counter] = elt.first;
        max_timings[counter]  = elt.second.count();
        mean_timings[counter] = elt.second.count();
        min_timings[counter]  = elt.second.count();
        counter++;
    }

    if (rankWorld == 0) {
        MPI_Reduce(MPI_IN_PLACE, max_timings.data(), max_timings.size(), MPI_DOUBLE, MPI_MAX, 0, comm);
        MPI_Reduce(MPI_IN_PLACE, min_timings.data(), min_timings.size(), MPI_DOUBLE, MPI_MIN, 0, comm);
        MPI_Reduce(MPI_IN_PLACE, mean_timings.data(), mean_timings.size(), MPI_DOUBLE, MPI_SUM, 0, comm);
    } else {
        MPI_Reduce(max_timings.data(), max_timings.data(), max_timings.size(), MPI_DOUBLE, MPI_MAX, 0, comm);
        MPI_Reduce(min_timings.data(), min_timings.data(), min_timings.size(), MPI_DOUBLE, MPI_MIN, 0, comm);
        MPI_Reduce(mean_timings.data(), mean_timings.data(), mean_timings.size(), MPI_DOUBLE, MPI_SUM, 0, comm);
    }
    if (rankWorld == 0) {
        for (auto &elt : mean_timings) {
            elt /= sizeWorld;
        }
        for (int i = 0; i < max_timings.size(); i++) {
            output << std::setw(output_size) << timing_names[i] + "_max" << max_timings[i] << " second(s)\n";
            output << std::setw(output_size) << timing_names[i] + "_mean" << mean_timings[i] << " second(s)\n";
            output << std::setw(output_size) << timing_names[i] + "_min" << min_timings[i] << " second(s)\n";
        }
    }

    os << output.str();
}
} // namespace htool

#endif
