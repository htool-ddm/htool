#ifndef HTOOL_HMATRIX_LINALG_OMP_TASK_ADD_HMATRIX_MATRIX_PRODUCT_ROW_MAJOR_HPP
#define HTOOL_HMATRIX_LINALG_OMP_TASK_ADD_HMATRIX_MATRIX_PRODUCT_ROW_MAJOR_HPP

#include "../../../misc/logger.hpp"
#include "../../../wrappers/wrapper_blas.hpp"
#include "../../hmatrix.hpp"
#include "../../task_dependencies.hpp"
#include "../seq/add_hmatrix_matrix_product_row_major.hpp"
#include <string>
#include <vector>

// to remove warning from depend(iterator(it = 0 : read_deps_size), in : *read_deps[it])
#if defined(__clang__)
#elif defined(__GNUC__) || defined(__GNUG__)
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wuseless-cast"
#endif

namespace htool {

// C = beta C + alpha op(A) B, with B and C stored row-major with mu columns, like sequential_internal_add_hmatrix_matrix_product_row_major,
// with one task per node of L0. The tasks depend on the first coefficient of the blocks of rows of B and C given by in_L0 and out_L0,
// cuts of the input and output cluster trees of op(A): one dependence per matrix and block, so that reading B does not wait for writing C,
// and successive task-based calls on the same matrix with the same cut are ordered. They also depend on the node of L0 they read.
// C is only complete after a taskwait.
template <typename CoefficientPrecision, typename CoordinatePrecision = underlying_type<CoefficientPrecision>>
void task_based_internal_add_hmatrix_matrix_product_row_major(char transa, CoefficientPrecision alpha, const HMatrix<CoefficientPrecision, CoordinatePrecision> &A, const CoefficientPrecision *B, CoefficientPrecision beta, CoefficientPrecision *C, int mu, const std::vector<const HMatrix<CoefficientPrecision, CoordinatePrecision> *> &L0, const std::vector<const Cluster<CoordinatePrecision> *> &in_L0, const std::vector<const Cluster<CoordinatePrecision> *> &out_L0) {
    if ((transa == 'T' && A.get_symmetry_for_leaves() == 'H')
        || (transa == 'C' && A.get_symmetry_for_leaves() == 'S')) {
        htool::Logger::get_instance().log(LogLevel::ERROR, "Operation is not supported (transa=" + std::string(1, transa) + " with " + A.get_symmetry_for_leaves() + ")"); // LCOV_EXCL_LINE
    }

    auto get_output_cluster{&HMatrix<CoefficientPrecision, CoordinatePrecision>::get_target_cluster};
    auto get_input_cluster{&HMatrix<CoefficientPrecision, CoordinatePrecision>::get_source_cluster};
    int local_input_offset  = A.get_source_cluster().get_offset();
    int local_output_offset = A.get_target_cluster().get_offset();
    char sym                = A.get_symmetry();
    char trans_sym          = (A.get_symmetry_for_leaves() == 'S') ? 'T' : 'C';
    if (transa != 'N') {
        get_input_cluster   = &HMatrix<CoefficientPrecision, CoordinatePrecision>::get_target_cluster;
        get_output_cluster  = &HMatrix<CoefficientPrecision, CoordinatePrecision>::get_source_cluster;
        local_input_offset  = A.get_target_cluster().get_offset();
        local_output_offset = A.get_source_cluster().get_offset();
        trans_sym           = 'N';
    }

    if (beta != CoefficientPrecision(1)) {
        for (auto out_L0_node : out_L0) {
            CoefficientPrecision *C_block = C + (out_L0_node->get_offset() - local_output_offset) * mu;
            int C_block_size              = out_L0_node->get_size() * mu;
#if defined(_OPENMP)
#    pragma omp task default(none) firstprivate(C_block, C_block_size, beta) depend(inout : *C_block)
#endif
            {
                int incx(1);
                Blas<CoefficientPrecision>::scal(&C_block_size, &beta, C_block, &incx);
            }
        }
    }

    for (auto L0_node : L0) {
        // In the symmetric case, an off-diagonal node is also applied transposed, from B on its output cluster to C on its input cluster:
        // in a separate task, so that each task writes in the blocks of only one cluster.
        bool is_off_diagonal_symmetric = sym != 'N' && (*L0_node.*get_input_cluster)().get_offset() != (*L0_node.*get_output_cluster)().get_offset();
        for (bool transposed : {false, true}) {
            if (transposed && !is_off_diagonal_symmetric) {
                continue;
            }
            char op                                   = transposed ? trans_sym : transa;
            const Cluster<CoordinatePrecision> &read  = transposed ? (*L0_node.*get_output_cluster)() : (*L0_node.*get_input_cluster)();
            const Cluster<CoordinatePrecision> &write = transposed ? (*L0_node.*get_input_cluster)() : (*L0_node.*get_output_cluster)();
            const CoefficientPrecision *B_node        = B + (read.get_offset() - local_input_offset) * mu;
            CoefficientPrecision *C_node              = C + (write.get_offset() - local_output_offset) * mu;

#if defined(_OPENMP)
            std::vector<CoefficientPrecision *> write_blocks;
            std::vector<const CoefficientPrecision *> read_blocks;
            for (auto cluster : enumerate_dependences(write, out_L0)) {
                write_blocks.push_back(C + (cluster->get_offset() - local_output_offset) * mu);
            }
            for (auto cluster : enumerate_dependences(read, in_L0)) {
                read_blocks.push_back(B + (cluster->get_offset() - local_input_offset) * mu);
            }
            auto write_deps_size = write_blocks.size();
            auto read_deps_size  = read_blocks.size();
#    pragma omp task default(none)                                                      \
        firstprivate(op, alpha, B_node, C_node, mu, L0_node, write_blocks, read_blocks) \
        depend(iterator(it = 0 : write_deps_size), inout : *write_blocks[it])           \
        depend(iterator(it = 0 : read_deps_size), in : *read_blocks[it])                \
        depend(in : *L0_node)
#else
            (void)in_L0;
#endif
            {
                sequential_internal_add_hmatrix_matrix_product_row_major(op, 'N', alpha, *L0_node, B_node, CoefficientPrecision(1), C_node, mu);
            }
        }
    }
}

} // namespace htool

#if defined(__clang__)
#elif defined(__GNUC__) || defined(__GNUG__)
#    pragma GCC diagnostic pop
#endif
#endif
