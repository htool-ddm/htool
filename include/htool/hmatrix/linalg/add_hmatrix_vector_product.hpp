#ifndef HTOOL_HMATRIX_LINALG_ADD_HMATRIX_VECTOR_PRODUCT_HPP
#define HTOOL_HMATRIX_LINALG_ADD_HMATRIX_VECTOR_PRODUCT_HPP

#include "../../matrix/linalg/add_matrix_vector_product.hpp" // for add_mat...
#include "../../misc/misc.hpp"                               // for underly...
#include "../../wrappers/wrapper_blas.hpp"                   // for Blas
#include "../execution_policies.hpp"
#include "../hmatrix.hpp"                               // for HMatrix
#include "../lrmat/linalg/add_lrmat_vector_product.hpp" // for add_lrm...
#include <algorithm>                                    // for transform
#include <complex>                                      // for complex
#include <cstddef>
#include <omp.h>
#include <vector> // for vector

namespace htool {

template <typename CoefficientPrecision, typename CoordinatePrecision = underlying_type<CoefficientPrecision>>
void internal_add_hmatrix_vector_product(char trans, CoefficientPrecision alpha, const HMatrix<CoefficientPrecision, CoordinatePrecision> &A, const CoefficientPrecision *in, CoefficientPrecision beta, CoefficientPrecision *out) {
    switch (A.get_storage_type()) {
    case HMatrix<CoefficientPrecision, CoordinatePrecision>::StorageType::Dense:
        if (A.get_symmetry() == 'N') {
            add_matrix_vector_product(trans, alpha, *A.get_dense_data(), in, beta, out);
        } else if (A.get_symmetry() == 'S') {
            add_symmetric_matrix_vector_product(A.get_UPLO(), alpha, *A.get_dense_data(), in, beta, out);
        }
        break;
    case HMatrix<CoefficientPrecision, CoordinatePrecision>::StorageType::LowRank:
        add_lrmat_vector_product(trans, alpha, *A.get_low_rank_data(), in, beta, out);
        break;
    default:
        sequential_internal_add_hmatrix_vector_product(trans, alpha, A, in, beta, out);
        break;
    }
}

template <typename CoefficientPrecision, typename CoordinatePrecision = underlying_type<CoefficientPrecision>>
void internal_add_hmatrix_vector_product(char trans, std::complex<CoefficientPrecision> alpha, const HMatrix<std::complex<CoefficientPrecision>, CoordinatePrecision> &A, const std::complex<CoefficientPrecision> *in, std::complex<CoefficientPrecision> beta, std::complex<CoefficientPrecision> *out) {
    switch (A.get_storage_type()) {
    case HMatrix<std::complex<CoefficientPrecision>, CoordinatePrecision>::StorageType::Dense:
        if (A.get_symmetry() == 'N') {
            add_matrix_vector_product(trans, alpha, *A.get_dense_data(), in, beta, out);
        } else if (A.get_symmetry() == 'S') {
            add_symmetric_matrix_vector_product(A.get_UPLO(), alpha, *A.get_dense_data(), in, beta, out);
        } else if (A.get_symmetry() == 'H') {
            add_hermitian_matrix_vector_product(A.get_UPLO(), alpha, *A.get_dense_data(), in, beta, out);
        }
        break;
    case HMatrix<std::complex<CoefficientPrecision>, CoordinatePrecision>::StorageType::LowRank:
        add_lrmat_vector_product(trans, alpha, *A.get_low_rank_data(), in, beta, out);
        break;
    default:
        sequential_internal_add_hmatrix_vector_product(trans, alpha, A, in, beta, out);
        break;
    }
}

template <typename CoefficientPrecision, typename CoordinatePrecision = underlying_type<CoefficientPrecision>>
void sequential_internal_add_hmatrix_vector_product(char trans, CoefficientPrecision alpha, const HMatrix<CoefficientPrecision, CoordinatePrecision> &A, const CoefficientPrecision *in, CoefficientPrecision beta, CoefficientPrecision *out) {

    if ((trans == 'T' && A.get_symmetry_for_leaves() == 'H')
        || (trans == 'C' && A.get_symmetry_for_leaves() == 'S')) {
        htool::Logger::get_instance().log(LogLevel::ERROR, "Operation is not supported (trans=" + std::string(1, trans) + " with " + A.get_symmetry_for_leaves() + ")"); // LCOV_EXCL_LINE
    }

    int out_size(A.get_target_cluster().get_size());
    auto get_output_cluster{&HMatrix<CoefficientPrecision, CoordinatePrecision>::get_target_cluster};
    auto get_input_cluster{&HMatrix<CoefficientPrecision, CoordinatePrecision>::get_source_cluster};
    int local_input_offset  = A.get_source_cluster().get_offset();
    int local_output_offset = A.get_target_cluster().get_offset();
    char trans_sym          = (A.get_symmetry_for_leaves() == 'S') ? 'T' : 'C';
    std::vector<const HMatrix<CoefficientPrecision, CoordinatePrecision> *> leaves;
    std::vector<const HMatrix<CoefficientPrecision, CoordinatePrecision> *> leaves_for_symmetry;
    std::tie(leaves, leaves_for_symmetry) = get_leaves_from(A); // C++17 structured binding

    if (trans != 'N') {
        out_size            = A.get_source_cluster().get_size();
        get_input_cluster   = &HMatrix<CoefficientPrecision, CoordinatePrecision>::get_target_cluster;
        get_output_cluster  = &HMatrix<CoefficientPrecision, CoordinatePrecision>::get_source_cluster;
        local_input_offset  = A.get_target_cluster().get_offset();
        local_output_offset = A.get_source_cluster().get_offset();
        trans_sym           = 'N';
    }

    if (CoefficientPrecision(beta) != CoefficientPrecision(1)) {
        int incx = 1;
        Blas<CoefficientPrecision>::scal(&out_size, &beta, out, &incx);
    }

    // Contribution champ lointain
    std::vector<CoefficientPrecision> temp(out_size, 0);
    for (auto &leaf : leaves) {
        int input_offset  = (leaf->*get_input_cluster)().get_offset();
        int output_offset = (leaf->*get_output_cluster)().get_offset();
        internal_add_hmatrix_vector_product(trans, alpha, *leaf, in + input_offset - local_input_offset, CoefficientPrecision(1), out + (output_offset - local_output_offset));
    }

    // Symmetry part of the diagonal part
    if (A.get_symmetry_for_leaves() != 'N') {
        for (auto &leaf_for_symmetry : leaves_for_symmetry) {
            int input_offset  = (leaf_for_symmetry->*get_input_cluster)().get_offset();
            int output_offset = (leaf_for_symmetry->*get_output_cluster)().get_offset();
            internal_add_hmatrix_vector_product(trans_sym, alpha, *leaf_for_symmetry, in + output_offset - local_input_offset, CoefficientPrecision(1), out + (input_offset - local_output_offset));
        }
    }
}

template <typename CoefficientPrecision, typename CoordinatePrecision = underlying_type<CoefficientPrecision>>
struct OpenMPAddHMatrixVectorProductCache {
    std::vector<const HMatrix<CoefficientPrecision, CoordinatePrecision> *> leaves;
    std::vector<std::pair<size_t, size_t>> work_load;
};

template <typename CoefficientPrecision, typename CoordinatePrecision = underlying_type<CoefficientPrecision>>
OpenMPAddHMatrixVectorProductCache<CoefficientPrecision, CoordinatePrecision> set_openmp_internal_add_hmatrix_vector_product(const HMatrix<CoefficientPrecision, CoordinatePrecision> &A) {
    OpenMPAddHMatrixVectorProductCache<CoefficientPrecision, CoordinatePrecision> cache;

    std::vector<const HMatrix<CoefficientPrecision, CoordinatePrecision> *> leaves_for_symmetry;
    std::tie(cache.leaves, leaves_for_symmetry) = get_leaves_from(A);
    size_t number_of_leaves                     = cache.leaves.size();
    int number_of_threads                       = omp_get_max_threads();
    cache.work_load.resize(number_of_threads);

    std::vector<size_t> accumulated_cost(number_of_leaves + 1);
    accumulated_cost[0] = 0;
    for (size_t p = 0; p < number_of_leaves; p++) {
        accumulated_cost[p + 1] = accumulated_cost[p];
        if (cache.leaves[p]->is_dense()) {
            accumulated_cost[p + 1] += cache.leaves[p]->nb_rows() * cache.leaves[p]->nb_cols();
        } else if (cache.leaves[p]->is_low_rank()) {
            accumulated_cost[p + 1] += cache.leaves[p]->get_rank() * (cache.leaves[p]->nb_rows() + cache.leaves[p]->nb_cols());
        }
    }
    size_t total_cost    = accumulated_cost[number_of_leaves];
    size_t current_index = 0;
    for (int t = 0; t < number_of_threads; t++) {
        size_t start = current_index;

        if (t == number_of_threads - 1) {
            current_index = number_of_leaves; // last thread takes all that's left
        } else {
            size_t target = total_cost * (t + 1) / number_of_threads;
            while (current_index < number_of_leaves && accumulated_cost[current_index] < target) {
                current_index++;
            }
            // don't starve the remaining threads of leaves
            size_t remaining_threads = number_of_threads - t - 1;
            if (number_of_leaves - current_index < remaining_threads) {
                current_index = number_of_leaves - remaining_threads;
            }
        }

        cache.work_load[t] = {start, current_index - start};
    }
    for (int p = 0; p < number_of_threads; p++) {
        std::cout << p << " " << (accumulated_cost[cache.work_load[p].second + cache.work_load[p].first] - accumulated_cost[cache.work_load[p].first]) / (double)total_cost << "\n";
    }
    return cache;
}

template <typename CoefficientPrecision, typename CoordinatePrecision = underlying_type<CoefficientPrecision>>
void openmp_internal_add_hmatrix_vector_product(char trans, CoefficientPrecision alpha, const HMatrix<CoefficientPrecision, CoordinatePrecision> &A, const CoefficientPrecision *in, CoefficientPrecision beta, CoefficientPrecision *out, const OpenMPAddHMatrixVectorProductCache<CoefficientPrecision, CoordinatePrecision> &cache) {

    const auto &leaves = cache.leaves;
    // const auto &leaves_for_symmetry = cache.leaves_for_symmetry; // see note below

    if ((trans == 'T' && A.get_symmetry_for_leaves() == 'H')
        || (trans == 'C' && A.get_symmetry_for_leaves() == 'S')) {
        htool::Logger::get_instance().log(LogLevel::ERROR, "Operation is not supported (trans=" + std::string(1, trans) + " with " + A.get_symmetry_for_leaves() + ")"); // LCOV_EXCL_LINE
    }

    int out_size(A.get_target_cluster().get_size());
    auto get_output_cluster{&HMatrix<CoefficientPrecision, CoordinatePrecision>::get_target_cluster};
    auto get_input_cluster{&HMatrix<CoefficientPrecision, CoordinatePrecision>::get_source_cluster};
    int local_input_offset  = A.get_source_cluster().get_offset();
    int local_output_offset = A.get_target_cluster().get_offset();
    char trans_sym          = (A.get_symmetry_for_leaves() == 'S') ? 'T' : 'C';
    if (trans != 'N') {
        out_size            = A.get_source_cluster().get_size();
        get_input_cluster   = &HMatrix<CoefficientPrecision, CoordinatePrecision>::get_target_cluster;
        get_output_cluster  = &HMatrix<CoefficientPrecision, CoordinatePrecision>::get_source_cluster;
        local_input_offset  = A.get_target_cluster().get_offset();
        local_output_offset = A.get_source_cluster().get_offset();
        trans_sym           = 'N';
    }

    int incx(1), incy(1);
    if (CoefficientPrecision(beta) != CoefficientPrecision(1)) {
        Blas<CoefficientPrecision>::scal(&out_size, &beta, out, &incx);
    }

    int number_of_threads = static_cast<int>(cache.work_load.size());

    // One buffer per thread, no shared writes during the leaf loop
    std::vector<std::vector<CoefficientPrecision>> temp_all(number_of_threads, std::vector<CoefficientPrecision>(out_size, CoefficientPrecision(0)));

#if defined(_OPENMP)
#    pragma omp parallel num_threads(number_of_threads)
#endif
    {
        // std::vector<CoefficientPrecision> temp(out_size, 0);

        // Far-field contribution: each thread reads its precomputed (offset, size)
        // instead of relying on the runtime scheduler
#if defined(_OPENMP)
        int tid       = omp_get_thread_num();
        size_t offset = cache.work_load[tid].first;
        size_t count  = cache.work_load[tid].second;
#else
        size_t offset = 0;
        size_t count  = leaves.size();
#endif
        auto &temp = temp_all[tid];
        for (size_t b = offset; b < offset + count; b++) {
            int input_offset  = (leaves[b]->*get_input_cluster)().get_offset();
            int output_offset = (leaves[b]->*get_output_cluster)().get_offset();
            internal_add_hmatrix_vector_product(trans, CoefficientPrecision(1), *leaves[b], in + input_offset - local_input_offset, CoefficientPrecision(1), temp.data() + (output_offset - local_output_offset));
        }

        //         // Symmetry part of the diagonal part -- left on dynamic scheduling for now
        //         if (A.get_symmetry_for_leaves() != 'N') {
        // #if defined(_OPENMP)
        // #    pragma omp for schedule(guided) nowait
        // #endif
        //             for (int b = 0; b < static_cast<int>(leaves_for_symmetry.size()); b++) {
        //                 int input_offset  = (leaves_for_symmetry[b]->*get_input_cluster)().get_offset();
        //                 int output_offset = (leaves_for_symmetry[b]->*get_output_cluster)().get_offset();
        //                 internal_add_hmatrix_vector_product(trans_sym, CoefficientPrecision(1), *leaves_for_symmetry[b], in + output_offset - local_input_offset, CoefficientPrecision(1), temp.data() + (input_offset - local_output_offset));
        //             }
        //         }

        // #if defined(_OPENMP)
        // #    pragma omp critical
        // #endif
        //         Blas<CoefficientPrecision>::axpy(&out_size, &alpha, temp.data(), &incx, out, &incy);
    }
    // Parallel reduction across threads' buffers -- replaces the critical axpy
#if defined(_OPENMP)
#    pragma omp parallel for num_threads(number_of_threads)
#endif
    for (int i = 0; i < out_size; i++) {
        CoefficientPrecision sum(0);
        for (int t = 0; t < number_of_threads; t++) {
            sum += temp_all[t][i];
        }
        out[i] += alpha * sum;
    }
}

template <typename CoefficientPrecision, typename CoordinatePrecision = underlying_type<CoefficientPrecision>>
void openmp_internal_add_hmatrix_vector_product(char trans, CoefficientPrecision alpha, const HMatrix<CoefficientPrecision, CoordinatePrecision> &A, const CoefficientPrecision *in, CoefficientPrecision beta, CoefficientPrecision *out) {
    std::vector<const HMatrix<CoefficientPrecision, CoordinatePrecision> *> leaves;
    std::vector<const HMatrix<CoefficientPrecision, CoordinatePrecision> *> leaves_for_symmetry;
    std::tie(leaves, leaves_for_symmetry) = get_leaves_from(A); // C++17 structured binding

    if ((trans == 'T' && A.get_symmetry_for_leaves() == 'H')
        || (trans == 'C' && A.get_symmetry_for_leaves() == 'S')) {
        htool::Logger::get_instance().log(LogLevel::ERROR, "Operation is not supported (trans=" + std::string(1, trans) + " with " + A.get_symmetry_for_leaves() + ")"); // LCOV_EXCL_LINE
    }

    int out_size(A.get_target_cluster().get_size());
    auto get_output_cluster{&HMatrix<CoefficientPrecision, CoordinatePrecision>::get_target_cluster};
    auto get_input_cluster{&HMatrix<CoefficientPrecision, CoordinatePrecision>::get_source_cluster};
    int local_input_offset  = A.get_source_cluster().get_offset();
    int local_output_offset = A.get_target_cluster().get_offset();
    char trans_sym          = (A.get_symmetry_for_leaves() == 'S') ? 'T' : 'C';

    if (trans != 'N') {
        out_size            = A.get_source_cluster().get_size();
        get_input_cluster   = &HMatrix<CoefficientPrecision, CoordinatePrecision>::get_target_cluster;
        get_output_cluster  = &HMatrix<CoefficientPrecision, CoordinatePrecision>::get_source_cluster;
        local_input_offset  = A.get_target_cluster().get_offset();
        local_output_offset = A.get_source_cluster().get_offset();
        trans_sym           = 'N';
    }

    int incx(1), incy(1);
    if (CoefficientPrecision(beta) != CoefficientPrecision(1)) {
        Blas<CoefficientPrecision>::scal(&out_size, &beta, out, &incx);
    }

    std::chrono::steady_clock::time_point start, end;
    std::chrono::duration<double> duration;
    start = std::chrono::steady_clock::now();
    // duration = end - start;
    // std::cout << "leafs" << duration.count() << std::endl;

// Contribution champ lointain
#if defined(_OPENMP)
#    pragma omp parallel
#endif
    {
        std::vector<CoefficientPrecision> temp(out_size, 0);
#if defined(_OPENMP)
#    pragma omp for schedule(guided) nowait
#endif
        for (int b = 0; b < leaves.size(); b++) {
            int input_offset  = (leaves[b]->*get_input_cluster)().get_offset();
            int output_offset = (leaves[b]->*get_output_cluster)().get_offset();
            internal_add_hmatrix_vector_product(trans, CoefficientPrecision(1), *leaves[b], in + input_offset - local_input_offset, CoefficientPrecision(1), temp.data() + (output_offset - local_output_offset));
        }

        // Symmetry part of the diagonal part
        if (A.get_symmetry_for_leaves() != 'N') {
#if defined(_OPENMP)
#    pragma omp for schedule(guided) nowait
#endif
            for (int b = 0; b < leaves_for_symmetry.size(); b++) {
                int input_offset  = (leaves_for_symmetry[b]->*get_input_cluster)().get_offset();
                int output_offset = (leaves_for_symmetry[b]->*get_output_cluster)().get_offset();
                internal_add_hmatrix_vector_product(trans_sym, CoefficientPrecision(1), *leaves_for_symmetry[b], in + output_offset - local_input_offset, CoefficientPrecision(1), temp.data() + (input_offset - local_output_offset));
            }
        }

#if defined(_OPENMP)
#    pragma omp critical
#endif
        Blas<CoefficientPrecision>::axpy(&out_size, &alpha, temp.data(), &incx, out, &incy);
    }
    end      = std::chrono::steady_clock::now();
    duration = end - start;
    std::cout << "timing " << duration.count() << std::endl;
}

template <typename ExecutionPolicy, typename CoefficientPrecision, typename CoordinatePrecision = underlying_type<CoefficientPrecision>>
void add_hmatrix_vector_product(ExecutionPolicy &&, char trans, CoefficientPrecision alpha, const HMatrix<CoefficientPrecision, CoordinatePrecision> &A, const CoefficientPrecision *in, CoefficientPrecision beta, CoefficientPrecision *out, CoefficientPrecision *buffer = nullptr) {
    auto &source_cluster = A.get_source_cluster();
    auto &target_cluster = A.get_target_cluster();
    std::vector<CoefficientPrecision> tmp(buffer == nullptr ? target_cluster.get_size() + source_cluster.get_size() : 0, 0);
    CoefficientPrecision *buffer_ptr = buffer == nullptr ? tmp.data() : buffer;
    user_to_cluster(source_cluster, in, buffer_ptr);
    user_to_cluster(target_cluster, out, buffer_ptr + source_cluster.get_size());

#if __cplusplus >= 201703L
    if constexpr (is_execution_policy_v<std::decay_t<ExecutionPolicy>>) {
        if constexpr (std::is_same_v<std::decay_t<ExecutionPolicy>, exec_compat::parallel_policy>) {
            openmp_internal_add_hmatrix_vector_product(trans, alpha, A, buffer_ptr, beta, buffer_ptr + source_cluster.get_size());
        } else if constexpr (std::is_same_v<std::decay_t<ExecutionPolicy>, exec_compat::sequenced_policy>) {
            sequential_internal_add_hmatrix_vector_product(trans, alpha, A, buffer_ptr, beta, buffer_ptr + source_cluster.get_size());
        } else {
            static_assert(std::is_same_v<std::decay_t<ExecutionPolicy>, exec_compat::sequenced_policy> || std::is_same_v<std::decay_t<ExecutionPolicy>, exec_compat::parallel_policy>, "Invalid execution policy for add_hmatrix_vector_product.");
        }
    } else {
        static_assert(is_execution_policy_v<std::decay_t<ExecutionPolicy>>, "Invalid execution policy for add_hmatrix_vector_product.");
    }
#else
    sequential_internal_add_hmatrix_vector_product(trans, alpha, A, buffer_ptr, beta, buffer_ptr + source_cluster.get_size());
#endif
    cluster_to_user(target_cluster, buffer_ptr + source_cluster.get_size(), out);
}

template <typename CoefficientPrecision, typename CoordinatePrecision = underlying_type<CoefficientPrecision>>
void add_hmatrix_vector_product(char trans, CoefficientPrecision alpha, const HMatrix<CoefficientPrecision, CoordinatePrecision> &A, const CoefficientPrecision *in, CoefficientPrecision beta, CoefficientPrecision *out, CoefficientPrecision *buffer = nullptr) {
#if __cplusplus >= 201703L
    add_hmatrix_vector_product(exec_compat::seq, trans, alpha, A, in, beta, out, buffer);
#else
    add_hmatrix_vector_product(nullptr, trans, alpha, A, in, beta, out, buffer);
#endif
}

} // namespace htool

#endif
