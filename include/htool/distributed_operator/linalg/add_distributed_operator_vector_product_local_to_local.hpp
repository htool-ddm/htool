#ifndef HTOOL_DISTRIBUTED_OPERATOR_LINALG_ADD_DISTRIBUTED_OPERATOR_VECTOR_PRODUCT_LOCAL_TO_LOCAL_HPP
#define HTOOL_DISTRIBUTED_OPERATOR_LINALG_ADD_DISTRIBUTED_OPERATOR_VECTOR_PRODUCT_LOCAL_TO_LOCAL_HPP

#include "../distributed_operator.hpp"
#include "utility.hpp"

namespace htool {

/// @brief
/// @tparam CoefficientPrecision
/// @param trans
/// @param alpha
/// @param A
/// @param in
/// @param beta
/// @param out
/// @param work Optional buffer of size (N if trans=='N', N+n*sizeWorld otherwise). Used if different of nullptr.
template <typename CoefficientPrecision>
void internal_add_distributed_operator_vector_product_local_to_local(char trans, CoefficientPrecision alpha, const DistributedOperator<CoefficientPrecision> &A, const CoefficientPrecision *const in, CoefficientPrecision beta, CoefficientPrecision *const out, CoefficientPrecision *work) {
    auto &input_partition           = trans == 'N' ? A.get_source_partition() : A.get_target_partition();
    auto &output_partition          = trans == 'N' ? A.get_target_partition() : A.get_source_partition();
    auto &global_to_local_operators = A.get_global_to_local_operators();
    auto &local_to_local_operators  = A.get_local_to_local_operators();
    bool apply_beta                 = true;

    if (local_to_local_operators.size() > 0) {
        for (auto &local_operator : local_to_local_operators) {
            local_operator->add_vector_product(trans, alpha, in, apply_beta ? beta : CoefficientPrecision(1), out);
            apply_beta = false;
        }
    }

    if (global_to_local_operators.size() > 0) {
        std::vector<CoefficientPrecision> buffer;
        const CoefficientPrecision *input_ptr;
        CoefficientPrecision *output_ptr;
        int sizeWorld, rankWorld;
        if (trans == 'N') {
            buffer.resize(work == nullptr ? input_partition.get_global_size() : 0, 0);
            input_ptr  = (work == nullptr) ? buffer.data() : work;
            output_ptr = out;

            // Local to global
            local_to_global(input_partition, in, (work == nullptr) ? buffer.data() : work, 1, A.get_comm());
        } else {
            MPI_Comm_rank(A.get_comm(), &rankWorld);
            MPI_Comm_size(A.get_comm(), &sizeWorld);
            buffer.resize(work == nullptr ? output_partition.get_global_size() + output_partition.get_size_of_partition(rankWorld) * sizeWorld : 0, 0);
            input_ptr  = in;
            output_ptr = (work == nullptr) ? buffer.data() : work;
            std::fill_n(output_ptr, output_partition.get_global_size() + output_partition.get_size_of_partition(rankWorld) * sizeWorld, CoefficientPrecision(0));
            apply_beta = false;
        }

        // Product
        for (auto &local_operator : A.get_global_to_local_operators()) {
            local_operator->add_vector_product(trans, alpha, input_ptr, apply_beta ? beta : CoefficientPrecision(1), output_ptr);
        }

        if (trans != 'N') {
            std::vector<int> scounts(sizeWorld), rcounts(sizeWorld);
            std::vector<int> sdispls(sizeWorld), rdispls(sizeWorld);
            CoefficientPrecision *rbuf = output_ptr + output_partition.get_global_size();

            sdispls[0] = 0;
            rdispls[0] = 0;

            for (int i = 0; i < sizeWorld; i++) {
                scounts[i] = output_partition.get_size_of_partition(i);
                rcounts[i] = output_partition.get_size_of_partition(rankWorld);
                if (i > 0) {
                    sdispls[i] = sdispls[i - 1] + scounts[i - 1];
                    rdispls[i] = rdispls[i - 1] + rcounts[i - 1];
                }
            }

            MPI_Alltoallv(output_ptr, &(scounts[0]), &(sdispls[0]), wrapper_mpi<CoefficientPrecision>::mpi_type(), rbuf, &(rcounts[0]), &(rdispls[0]), wrapper_mpi<CoefficientPrecision>::mpi_type(), A.get_comm());

            int inc                  = 1;
            CoefficientPrecision one = 1;
            int local_size           = output_partition.get_size_of_partition(rankWorld);
            if (local_to_local_operators.size() == 0)
                Blas<CoefficientPrecision>::scal(&local_size, &beta, out, &inc);
            for (int i = 0; i < sizeWorld; i++) {
                Blas<CoefficientPrecision>::axpy(&local_size, &one, rbuf + rdispls[i], &inc, out, &inc);
            }
        }
    }
}

/// @brief
/// @tparam CoefficientPrecision
/// @param trans
/// @param alpha
/// @param A
/// @param in
/// @param beta
/// @param out
/// @param work Optional buffer of size (m+n+N if trans=='N', m+n+N+n*sizeWorld otherwise). Used if different of nullptr.
template <typename CoefficientPrecision>
void add_distributed_operator_vector_product_local_to_local(char trans, CoefficientPrecision alpha, const DistributedOperator<CoefficientPrecision> &A, const CoefficientPrecision *const in, CoefficientPrecision beta, CoefficientPrecision *const out, CoefficientPrecision *work) {
    int rankWorld;
    MPI_Comm_rank(A.get_comm(), &rankWorld);
    auto &input_partition  = trans == 'N' ? A.get_source_partition() : A.get_target_partition();
    auto &output_partition = trans == 'N' ? A.get_target_partition() : A.get_source_partition();
    int input_local_size   = input_partition.get_size_of_partition(rankWorld);
    int output_local_size  = output_partition.get_size_of_partition(rankWorld);
    std::vector<CoefficientPrecision> input_buffer(work == nullptr ? input_local_size : 0, 0);
    std::vector<CoefficientPrecision> output_buffer(work == nullptr ? output_local_size : 0, 0);
    CoefficientPrecision *input_buffer_ptr  = work == nullptr ? input_buffer.data() : work;
    CoefficientPrecision *output_buffer_ptr = work == nullptr ? output_buffer.data() : work + input_local_size;

    // Permutation
    input_partition.local_to_local_partition_numbering(rankWorld, in, input_buffer_ptr);
    if (beta != CoefficientPrecision(0)) {
        output_partition.local_to_local_partition_numbering(rankWorld, out, output_buffer_ptr);
    }

    // Product
    internal_add_distributed_operator_vector_product_local_to_local(trans, alpha, A, input_buffer_ptr, beta, output_buffer_ptr, work == nullptr ? nullptr : work + input_local_size + output_local_size);

    // Permutation
    output_partition.local_partition_to_local_numbering(rankWorld, output_buffer_ptr, out);
}
} // namespace htool

#endif
