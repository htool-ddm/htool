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
std::map<std::string, std::string> get_distributed_hmatrix_information(const HMatrix<CoefficientPrecision, CoordinatePrecision> &hmatrix, MPI_Comm comm) {
    std::map<std::string, std::string> distributed_information;
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

    // Print
    if (rankWorld == 0) {
        distributed_information["Target_size_max"]           = std::to_string(maxinfos[3]);
        distributed_information["Target_size_mean"]          = std::to_string(meaninfos[3]);
        distributed_information["Target_size_min"]           = std::to_string(mininfos[3]);
        distributed_information["Source_size_max"]           = std::to_string(maxinfos[4]);
        distributed_information["Source_size_mean"]          = std::to_string(meaninfos[4]);
        distributed_information["Source_size_min"]           = std::to_string(mininfos[4]);
        distributed_information["Dense_block_size_max"]      = std::to_string(maxinfos[0]);
        distributed_information["Dense_block_size_mean"]     = std::to_string(meaninfos[0]);
        distributed_information["Dense_block_size_min"]      = std::to_string(mininfos[0]);
        distributed_information["Low_rank_block_size_max"]   = std::to_string(maxinfos[1]);
        distributed_information["Low_rank_block_size_mean"]  = std::to_string(meaninfos[1]);
        distributed_information["Low_rank_block_size_min"]   = std::to_string(mininfos[1]);
        distributed_information["Rank_max"]                  = std::to_string(maxinfos[2]);
        distributed_information["Rank_mean"]                 = std::to_string(meaninfos[2]);
        distributed_information["Rank_min"]                  = std::to_string(mininfos[2]);
        distributed_information["Number_of_low_rank_blocks"] = std::to_string(nb_low_rank_blocks);
        distributed_information["Number_of_dense_blocks"]    = std::to_string(nb_dense_blocks);
        distributed_information["Compression_ratio"]         = std::to_string((total_size) / static_cast<double>(total_number_of_generated_coefficients));
        distributed_information["Space_saving"]              = std::to_string(1 - static_cast<double>(total_number_of_generated_coefficients) / (total_size));
        distributed_information["Number_of_MPI_tasks"]       = std::to_string(sizeWorld);
#if defined(_OPENMP)
        distributed_information["Number_of_threads_per_tasks"] = std::to_string(omp_get_max_threads());
        distributed_information["Number_of_procs"]             = std::to_string(sizeWorld * omp_get_max_threads());
#else
        distributed_information["Number_of_procs"] = std::to_string(sizeWorld);
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
            distributed_information[information_name[i]] = std::to_string(information_value[i]);
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
            distributed_information[timing_names[i] + "_max"]  = std::to_string(max_timings[i]) + " second(s)";
            distributed_information[timing_names[i] + "_mean"] = std::to_string(mean_timings[i]) + " second(s)";
            distributed_information[timing_names[i] + "_min"]  = std::to_string(min_timings[i]) + " second(s)";
        }
    }

    return distributed_information;
}

template <typename CoefficientPrecision, typename CoordinatePrecision>
void print_distributed_hmatrix_information(const HMatrix<CoefficientPrecision, CoordinatePrecision> &hmatrix, std::ostream &os, MPI_Comm comm) {
    auto distributed_information = get_distributed_hmatrix_information(hmatrix, comm);
    int rankWorld;
    MPI_Comm_rank(comm, &rankWorld);
    if (rankWorld == 0) {
        std::size_t output_size = 2 + std::max_element(std::begin(distributed_information), std::end(distributed_information), [](const auto &a, const auto &b) { return a.first.size() < b.first.size(); })->first.size();
        // save default formatting
        std::ios init(NULL);
        init.copyfmt(os);

        os << std::setfill('_') << std::left;
        os << "Distributed Hmatrix information\n";

        for (const auto &information : distributed_information) {
            os << std::setw(output_size) << information.first << information.second << "\n";
        }

        // restore default formatting
        os.copyfmt(init);
    }
}
} // namespace htool

#endif
