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
 * @brief Performs a task-based parallel addition of HMatrix-vector products to an output vector.
 *
 * This function computes the product of a hierarchical matrix (HMatrix) and an input vector,
 * scales the output vector by a given factor, and adds the result to the output vector.
 * The computation is performed in parallel using OpenMP tasks, with dependencies managed
 * to ensure correct updates to the output vector. The function supports symmetric matrices
 * and transposed operations.
 *
 * @tparam CoefficientPrecision Type used for matrix coefficients and vector elements.
 * @tparam CoordinatePrecision Type used for cluster coordinates.
 * @param trans Character specifying the operation: 'N' for normal, other values for transpose/conjugate.
 * @param alpha Scalar multiplier for the matrix-vector product.
 * @param A The hierarchical matrix (HMatrix) to be multiplied.
 * @param in Pointer to the input vector.
 * @param beta Scalar multiplier for the output vector.
 * @param out Pointer to the output vector to which the result is added.
 * @param L0 Vector of pointers to HMatrix nodes for task scheduling.
 * @param in_L0 Vector of pointers to input clusters corresponding to L0 nodes.
 * @param out_L0 Vector of pointers to output clusters corresponding to L0 nodes.
 */
template <typename CoefficientPrecision, typename CoordinatePrecision = CoefficientPrecision>
void task_based_internal_add_hmatrix_vector_product(char trans, CoefficientPrecision alpha, const HMatrix<CoefficientPrecision, CoordinatePrecision> &A, const CoefficientPrecision *in, CoefficientPrecision beta, CoefficientPrecision *out, const std::vector<HMatrix<CoefficientPrecision, CoordinatePrecision> *> &L0, const std::vector<const Cluster<CoordinatePrecision> *> &in_L0, const std::vector<const Cluster<CoordinatePrecision> *> &out_L0) {

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
        for (auto out_L0_node : out_L0) {

#if defined(_OPENMP) && !defined(HTOOL_WITH_PYTHON_INTERFACE)
#    pragma omp task default(none)           \
        firstprivate(out_L0_node, out, beta) \
        depend(inout : *out_L0_node)
// priority(max_prio - 1)
#endif // 'out' must be copied by the thread else stack-use-after-return error
            {
                int out_L0_node_offset = out_L0_node->get_offset();
                int out_L0_node_size   = out_L0_node->get_size();
                int incx(1);
                Blas<CoefficientPrecision>::scal(&out_L0_node_size, &beta, out + out_L0_node_offset, &incx);
            }
        }
    }

    for (auto L0_node : L0) {
        int input_offset  = (*L0_node.*get_input_cluster)().get_offset();
        int output_offset = (*L0_node.*get_output_cluster)().get_offset();

#if defined(_OPENMP) && !defined(HTOOL_WITH_PYTHON_INTERFACE)
        std::vector<const Cluster<CoordinatePrecision> *> write_cluster, read_cluster;
        write_cluster = enumerate_dependences((*L0_node.*get_output_cluster)(), out_L0);
        read_cluster  = enumerate_dependences((*L0_node.*get_input_cluster)(), in_L0);

        // concatenate write_cluster and read_cluster for the symetric case
        if (sym != 'N') {
            write_cluster.insert(write_cluster.end(), read_cluster.begin(), read_cluster.end());
            read_cluster.clear();
        }
        auto write_deps_size        = write_cluster.size();
        auto read_deps_cluster_size = read_cluster.size();
#    pragma omp task default(none)                                                                                                                                                       \
        firstprivate(sym, trans_sym, alpha, in, trans, out, write_deps_size, input_offset, output_offset, local_input_offset, local_output_offset, L0_node, write_cluster, read_cluster) \
        shared(std::cout)                                                                                                                                                                \
        depend(iterator(it = 0 : write_deps_size), inout : *write_cluster[it])                                                                                                           \
        depend(iterator(it = 0 : read_deps_cluster_size), in : *read_cluster[it])                                                                                                        \
        depend(in : *L0_node)
        // priority(max_prio)
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
