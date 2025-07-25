#ifndef HTOOL_TASK_BASED_HMATRIX_LINALG_ADD_HMATRIX_VECTOR_PRODUCT_HPP
#define HTOOL_TASK_BASED_HMATRIX_LINALG_ADD_HMATRIX_VECTOR_PRODUCT_HPP

// #include "../../matrix/linalg/add_matrix_vector_product.hpp"      // for add_mat...
// #include "../../misc/misc.hpp"                                    // for underly...
#include "../../wrappers/wrapper_blas.hpp" // for Blas
#include "../hmatrix.hpp"                  // for HMatrix
// #include "../lrmat/linalg/add_lrmat_vector_product.hpp"           // for add_lrm...
#include <algorithm> // for transform, max
// #include <complex>                                                // for complex
#include <htool/hmatrix/tree_builder/task_based_tree_builder.hpp> // for enumerate_dependence, find_l0...
#include <mutex>
#include <vector> // for vector

#include <unistd.h>

// to remove warning from depend(iterator(it = 0 : read_deps_size), in : *read_deps[it])
#if defined(__clang__)
#elif defined(__GNUC__) || defined(__GNUG__)
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wuseless-cast"
#endif

namespace htool {

/**
 * @brief Task-based internal add hmatrix vector product, i.e., C = beta*C + alpha*A*x, where A is a HMatrix, x is a vector, and C is a vector.
 *
 * @details
 * The function returns the result in the vector C.
 * The function takes as input the HMatrix A, the vector x, the scalar alpha, and the scalar beta.
 * The function also takes as input the vector L0, which is a vector of pointers to nodes of the tree of A, and the vectors in_L0 and out_L0, which are vectors of pointers to the cluster nodes of the input and output vectors x and C, respectively.
 * The function also takes as input a pointer to a mutex, which is used to synchronize the output of the tasks.
 * If the mutex is nullptr, the function does not use any synchronization.
 */
template <typename CoefficientPrecision, typename CoordinatePrecision = CoefficientPrecision>
void task_based_internal_add_hmatrix_vector_product(char trans, CoefficientPrecision alpha, const HMatrix<CoefficientPrecision, CoordinatePrecision> &A, const CoefficientPrecision *in, CoefficientPrecision beta, CoefficientPrecision *out, const std::vector<HMatrix<CoefficientPrecision, CoordinatePrecision> *> &L0, const std::vector<const Cluster<CoordinatePrecision> *> &in_L0, const std::vector<const Cluster<CoordinatePrecision> *> &out_L0, std::mutex *cout_mutex = nullptr) {

    auto get_output_cluster{&HMatrix<CoefficientPrecision, CoordinatePrecision>::get_target_cluster};
    auto get_input_cluster{&HMatrix<CoefficientPrecision, CoordinatePrecision>::get_source_cluster};
    char sym       = A.get_symmetry();
    char trans_sym = (A.get_symmetry_for_leaves() == 'S') ? 'T' : 'C';
    if (trans != 'N') {
        get_input_cluster  = &HMatrix<CoefficientPrecision, CoordinatePrecision>::get_target_cluster;
        get_output_cluster = &HMatrix<CoefficientPrecision, CoordinatePrecision>::get_source_cluster;
        trans_sym          = 'N';
    }

    int local_input_offset  = A.get_source_cluster().get_offset();
    int local_output_offset = A.get_target_cluster().get_offset();
    // int local_output_size   = A.get_target_cluster().get_size();

    // int max_prio = std::max(0, omp_get_max_task_priority());

    // Scale the output vector `out` with `beta` if `beta` is not equal to 1
    if (CoefficientPrecision(beta) != CoefficientPrecision(1)) {
        for (auto &out_L0_node : out_L0) {
            int out_L0_node_offset = out_L0_node->get_offset();
            int out_L0_node_size   = out_L0_node->get_size();

#if defined(_OPENMP) && !defined(HTOOL_WITH_PYTHON_INTERFACE)
#    pragma omp task default(none)                                                \
        firstprivate(out_L0_node_offset, out_L0_node_size, out, beta, cout_mutex) \
        shared(out_L0_node, std::cout)                                            \
        depend(inout : *out_L0_node)
            // priority(max_prio - 1)
#endif // 'out' must be copied by the thread else stack-use-after-return error
            {
                if (cout_mutex != nullptr) {
                    std::lock_guard<std::mutex> lock(*cout_mutex);
                    std::cout << "Scaling : out[" << out_L0_node_offset << " : " << out_L0_node_offset + out_L0_node_size << "], thread " << omp_get_thread_num() << "\n";
                }
                int incx(1);
                Blas<CoefficientPrecision>::scal(&out_L0_node_size, &beta, out + out_L0_node_offset, &incx);
            }
        }
    }

    for (auto &L0_node : L0) {
        int input_offset  = (*L0_node.*get_input_cluster)().get_offset();
        int output_offset = (*L0_node.*get_output_cluster)().get_offset();

        std::vector<const Cluster<CoordinatePrecision> *> write_cluster, read_cluster;
        write_cluster = enumerate_dependences((*L0_node.*get_output_cluster)(), out_L0);
        read_cluster  = enumerate_dependences((*L0_node.*get_input_cluster)(), in_L0);

        // concatenate write_cluster and read_cluster
        write_cluster.insert(write_cluster.end(), read_cluster.begin(), read_cluster.end());

        auto write_deps_size = write_cluster.size();
        // auto read_deps_cluster_size = read_cluster.size();

#if defined(_OPENMP) && !defined(HTOOL_WITH_PYTHON_INTERFACE)
#    pragma omp task default(none)                                                                                                                                      \
        firstprivate(sym, trans_sym, alpha, in, trans, out, write_deps_size, cout_mutex, input_offset, output_offset, local_input_offset, local_output_offset, L0_node) \
        shared(std::cout, write_cluster, read_cluster)                                                                                                                  \
        depend(iterator(it = 0 : write_deps_size), inout : *write_cluster[it])                                                                                          \
        depend(in : *L0_node)
        // priority(max_prio)
        // depend(iterator(it = 0 : read_deps_cluster_size), in : *read_cluster[it])
#endif // L0_node must be copied by the thread else heap-use-after-free error
        {
            sequential_internal_add_hmatrix_vector_product(trans, alpha, *L0_node, in + input_offset - local_input_offset, CoefficientPrecision(1), out + output_offset - local_output_offset);
            if (sym != 'N' && input_offset != output_offset) {
                sequential_internal_add_hmatrix_vector_product(trans_sym, alpha, *L0_node, in + output_offset - local_input_offset, CoefficientPrecision(1), out + input_offset - local_output_offset);
            }
        }
    }
} // end of task_based_internal_add_hmatrix_vector_product

} // namespace htool

#if defined(__clang__)
#elif defined(__GNUC__) || defined(__GNUG__)
#    pragma GCC diagnostic pop
#endif
#endif
