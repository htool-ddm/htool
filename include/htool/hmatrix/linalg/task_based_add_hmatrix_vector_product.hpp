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

namespace htool {

/**
 * \brief Task-based internal add HMatrix vector product function
 *
 * This function is a task-based version of the internal add HMatrix vector product function.
 * It uses OpenMP tasks to parallelize the computation of the product for each node of the tree.
 * The function takes the following parameters:
 *      - `trans`: a character indicating whether the product should be computed as \f$Ax\f$ or \f$A^Tx\f$
 *      - `alpha`: a scalar coefficient
 *      - `A`: a HMatrix object
 *      - `in`: the input vector
 *      - `beta`: a scalar coefficient
 *      - `out`: the output vector
 *      - `L0`: a vector of pointers to the nodes of the tree at level 0
 *      - `in_L0`: a vector of pointers to the input clusters at level 0
 *      - `out_L0`: a vector of pointers to the output clusters at level 0
 *      - `cout_mutex`: a pointer to a mutex object used for synchronization of the output (optional)
 *
 * The function first scales the output vector `out` with `beta` if `beta` is not equal to 1.
 * Then, it checks if the target cluster is in L0. If it is, the function computes the product of the HMatrix and the input vector for each node of the tree.
 * If the target cluster is not in L0, the function recursively calls itself for each child of the current HMatrix.
 *
 * The function uses OpenMP tasks to parallelize the computation of the product for each node of the tree.
 * The tasks are created with the `depend` clause to ensure that the tasks are executed in the correct order.
 * The function uses the `priority` clause to set the priority of the tasks.
 */
template <typename CoefficientPrecision, typename CoordinatePrecision = CoefficientPrecision>
void task_based_internal_add_hmatrix_vector_product(char trans, CoefficientPrecision alpha, const HMatrix<CoefficientPrecision, CoordinatePrecision> &A, const CoefficientPrecision *in, CoefficientPrecision beta, CoefficientPrecision *out, const std::vector<HMatrix<CoefficientPrecision, CoordinatePrecision> *> &L0, const std::vector<const Cluster<CoordinatePrecision> *> &in_L0, const std::vector<const Cluster<CoordinatePrecision> *> &out_L0, std::mutex *cout_mutex = nullptr) {

    // Todo : maybe its not efficient because of the recursive call
    auto get_output_cluster{&HMatrix<CoefficientPrecision, CoordinatePrecision>::get_target_cluster};
    auto get_input_cluster{&HMatrix<CoefficientPrecision, CoordinatePrecision>::get_source_cluster};
    if (trans != 'N') {
        get_input_cluster  = &HMatrix<CoefficientPrecision, CoordinatePrecision>::get_target_cluster;
        get_output_cluster = &HMatrix<CoefficientPrecision, CoordinatePrecision>::get_source_cluster;
    }

    const auto &target_cluster = (A.*get_output_cluster)();
    const auto &source_cluster = (A.*get_input_cluster)();
    int local_input_offset     = source_cluster.get_offset();
    int local_input_size       = source_cluster.get_size();
    int local_output_offset    = target_cluster.get_offset();
    int local_output_size      = target_cluster.get_size();

    int max_prio = omp_get_max_task_priority(); // == 0 if not specified by user at runtime with : OMP_MAX_TASK_PRIORITY=<value> ./program

    // Scale the output vector `out` with `beta` if `beta` is not equal to 1
    if (CoefficientPrecision(beta) != CoefficientPrecision(1)) {
        for (auto &out_L0_nodes : out_L0) {
            int out_L0_nodes_offset = out_L0_nodes->get_offset();
            int out_L0_nodes_size   = out_L0_nodes->get_size();

#if defined(_OPENMP) && !defined(HTOOL_WITH_PYTHON_INTERFACE)
#    pragma omp task default(none)                                            \
        firstprivate(out_L0_nodes_offset, out_L0_nodes_size, out, cout_mutex) \
        shared(beta, local_output_size, out_L0_nodes, std::cout)              \
        depend(inout : *out_L0_nodes)                                         \
        priority(std::max(0, max_prio - 1))
#endif
            {
                if (cout_mutex != nullptr) {
                    std::lock_guard<std::mutex> lock(*cout_mutex);
                    std::cout << "Scaling : out[" << out_L0_nodes_offset << " : " << out_L0_nodes_offset + out_L0_nodes_size << "], thread " << omp_get_thread_num() << "\n";
                    // std::cout << "out_L0_nodes = " << out_L0_nodes << "\n";
                }
                int incx(1);
                Blas<CoefficientPrecision>::scal(&out_L0_nodes_size, &beta, out + out_L0_nodes_offset, &incx);
            }
        }
    }

    // check if the target cluster is in L0
    bool is_local_out_in_L0 = false;
    for (auto &target_cluster_on_L0 : out_L0) {
        if (target_cluster == *target_cluster_on_L0) {
            is_local_out_in_L0 = true;
            break;
        }
    }

    if (is_local_out_in_L0 || A.is_leaf()) {
        std::vector<const HMatrix<CoefficientPrecision, CoordinatePrecision> *> read_dependences_hmatrix;
        read_dependences_hmatrix = enumerate_dependences(A, L0);
        std::vector<const Cluster<CoordinatePrecision> *> write_dependences, read_dependences_cluster;
        write_dependences        = enumerate_dependences(target_cluster, out_L0);
        read_dependences_cluster = enumerate_dependences(source_cluster, in_L0);

        auto write_deps_size        = write_dependences.size();
        auto read_deps_hmatrix_size = read_dependences_hmatrix.size();
        auto read_deps_cluster_size = read_dependences_cluster.size();

#if defined(_OPENMP) && !defined(HTOOL_WITH_PYTHON_INTERFACE)
#    pragma omp task default(none)                                                                                                                                                                                                                                                    \
        shared(A, alpha, in, local_output_offset, local_output_size, local_input_offset, local_input_size, trans, out, write_dependences, write_deps_size, read_deps_hmatrix_size, read_dependences_hmatrix, read_deps_cluster_size, read_dependences_cluster, cout_mutex, std::cout) \
        depend(iterator(it = 0 : write_deps_size), inout : *write_dependences[it])                                                                                                                                                                                                    \
        priority(std::max(0, max_prio))
        // depend(iterator(it = 0 : read_deps_cluster_size), in : *read_dependences_cluster[it]) // Same as Scaling dependences, hence introduce a superfluous dependence. Todo : find a better way => change enumerate_dependences to return a pair of (vector<const Hmatrix/Cluster *>, vector<const Cluster<T> *>)
        // depend(iterator(it = 0 : read_deps_hmatrix_size), in : *read_dependences_hmatrix[it]) // Same issue but between two product tasks.
#endif
        {
            if (cout_mutex != nullptr) {
                std::lock_guard<std::mutex> lock(*cout_mutex);
                std::cout << " Product write in : out[" << local_output_offset << " : " << local_output_offset + local_output_size << "] and read in : in[" << local_input_offset << " : " << local_input_offset + local_input_size << "], thread " << omp_get_thread_num() << "\n";
                // for (int i = 0; i < read_deps_cluster_size; i++) {
                //     std::cout << " read_dependences_cluster[" << i << "] = " << read_dependences_cluster[i] << "\n";
                // }
            }
            internal_add_hmatrix_vector_product(trans, alpha, A, in, CoefficientPrecision(1), out);
        }

    } else {
        // Recursively call the function for each child of the current HMatrix
        for (const auto &child : A.get_children()) {
            int input_offset  = (child.get()->*get_input_cluster)().get_offset();
            int output_offset = (child.get()->*get_output_cluster)().get_offset();

            task_based_internal_add_hmatrix_vector_product(trans, alpha, *child.get(), in + input_offset - local_input_offset, CoefficientPrecision(1), out + output_offset - local_output_offset, L0, in_L0, out_L0, cout_mutex);
        }
    } // end of if (is_local_out_in_L0 || A.is_leaf())

} // end of task_based_internal_add_hmatrix_vector_product

} // namespace htool

#endif
