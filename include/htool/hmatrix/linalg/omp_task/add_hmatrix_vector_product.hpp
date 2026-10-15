#ifndef HTOOL_HMATRIX_LINALG_OMP_TASK_ADD_HMATRIX_VECTOR_PRODUCT_HPP
#define HTOOL_HMATRIX_LINALG_OMP_TASK_ADD_HMATRIX_VECTOR_PRODUCT_HPP

// #include "../../matrix/linalg/add_matrix_vector_product.hpp"      // for add_mat...
// #include "../../misc/misc.hpp"                                    // for underly...
#include "../../../wrappers/wrapper_blas.hpp" // for Blas
#include "../../hmatrix.hpp"
#include "../seq/add_hmatrix_vector_product.hpp" // for HMatrix
// #include "../lrmat/linalg/add_lrmat_vector_product.hpp"           // for add_lrm...
#include <algorithm> // for transform, max
// #include <complex>                                                // for complex
#include <htool/hmatrix/task_dependencies.hpp> // for enumerate_dependence, find_l0...
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
void task_based_internal_add_hmatrix_vector_product(char trans, CoefficientPrecision alpha, const HMatrix<CoefficientPrecision, CoordinatePrecision> &A, const CoefficientPrecision *in, CoefficientPrecision beta, CoefficientPrecision *out, const std::vector<const HMatrix<CoefficientPrecision, CoordinatePrecision> *> &L0, const std::vector<const Cluster<CoordinatePrecision> *> &in_L0, const std::vector<const Cluster<CoordinatePrecision> *> &out_L0) {

    auto get_output_cluster{&HMatrix<CoefficientPrecision, CoordinatePrecision>::get_target_cluster};
    auto get_input_cluster{&HMatrix<CoefficientPrecision, CoordinatePrecision>::get_source_cluster};
    char sym                = A.get_symmetry();
    char trans_sym          = (A.get_symmetry_for_leaves() == 'S') ? 'T' : 'C';
    int local_input_offset  = A.get_source_cluster().get_offset();
    int local_output_offset = A.get_target_cluster().get_offset();
    if (trans != 'N') {
        get_input_cluster   = &HMatrix<CoefficientPrecision, CoordinatePrecision>::get_target_cluster;
        get_output_cluster  = &HMatrix<CoefficientPrecision, CoordinatePrecision>::get_source_cluster;
        local_input_offset  = A.get_target_cluster().get_offset();
        local_output_offset = A.get_source_cluster().get_offset();
        trans_sym           = 'N';
    }

    // The tasks depend on the first coefficient of the blocks of in and out given by in_L0 and out_L0: one dependence per vector and block,
    // so that reading in does not wait for writing out, and successive task-based calls on the same vector with the same cut are ordered.
    if (CoefficientPrecision(beta) != CoefficientPrecision(1)) {
        for (auto out_L0_node : out_L0) {
            CoefficientPrecision *out_block = out + out_L0_node->get_offset() - local_output_offset;
            int out_block_size              = out_L0_node->get_size();
#if defined(_OPENMP)
#    pragma omp task default(none) firstprivate(out_block, out_block_size, beta) depend(inout : *out_block)
#endif
            {
                int incx(1);
                Blas<CoefficientPrecision>::scal(&out_block_size, &beta, out_block, &incx);
            }
        }
    }

    for (auto L0_node : L0) {
        // In the symmetric case, an off-diagonal node is also applied transposed, from in on its output cluster to out on its input cluster:
        // in a separate task, so that each task writes in the blocks of only one cluster.
        bool is_off_diagonal_symmetric = sym != 'N' && (*L0_node.*get_input_cluster)().get_offset() != (*L0_node.*get_output_cluster)().get_offset();
        for (bool transposed : {false, true}) {
            if (transposed && !is_off_diagonal_symmetric) {
                continue;
            }
            char op                                   = transposed ? trans_sym : trans;
            const Cluster<CoordinatePrecision> &read  = transposed ? (*L0_node.*get_output_cluster)() : (*L0_node.*get_input_cluster)();
            const Cluster<CoordinatePrecision> &write = transposed ? (*L0_node.*get_input_cluster)() : (*L0_node.*get_output_cluster)();
            const CoefficientPrecision *in_node       = in + read.get_offset() - local_input_offset;
            CoefficientPrecision *out_node            = out + write.get_offset() - local_output_offset;

#if defined(_OPENMP)
            std::vector<CoefficientPrecision *> write_blocks;
            std::vector<const CoefficientPrecision *> read_blocks;
            for (auto cluster : enumerate_dependences(write, out_L0)) {
                write_blocks.push_back(out + cluster->get_offset() - local_output_offset);
            }
            for (auto cluster : enumerate_dependences(read, in_L0)) {
                read_blocks.push_back(in + cluster->get_offset() - local_input_offset);
            }
            auto write_deps_size = write_blocks.size();
            auto read_deps_size  = read_blocks.size();
#    pragma omp task default(none)                                                     \
        firstprivate(op, alpha, in_node, out_node, L0_node, write_blocks, read_blocks) \
        depend(iterator(it = 0 : write_deps_size), inout : *write_blocks[it])          \
        depend(iterator(it = 0 : read_deps_size), in : *read_blocks[it])               \
        depend(in : *L0_node)
#else
            (void)in_L0;
#endif // L0_node must be copied by the thread else heap-use-after-free error
            {
                sequential_internal_add_hmatrix_vector_product(op, alpha, *L0_node, in_node, CoefficientPrecision(1), out_node);
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
