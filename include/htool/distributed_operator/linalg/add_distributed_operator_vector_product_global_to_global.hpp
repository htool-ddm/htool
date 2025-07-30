#ifndef HTOOL_DISTRIBUTED_OPERATOR_LINALG_ADD_DISTRIBUTED_OPERATOR_VECTOR_PRODUCT_GLOBAL_TO_GLOBAL_HPP
#define HTOOL_DISTRIBUTED_OPERATOR_LINALG_ADD_DISTRIBUTED_OPERATOR_VECTOR_PRODUCT_GLOBAL_TO_GLOBAL_HPP

#include "../distributed_operator.hpp"

namespace htool {

/// @brief
/// @tparam CoefficientPrecision
/// @param trans
/// @param alpha
/// @param A
/// @param in
/// @param beta
/// @param out
/// @param work Optional buffer of size (m if trans=='N', N if trans!='N' and beta!=0, 0 otherwise). Used if different of nullptr.
template <typename CoefficientPrecision>
void internal_add_distributed_operator_vector_product_global_to_global(char trans, CoefficientPrecision alpha, const DistributedOperator<CoefficientPrecision> &A, const CoefficientPrecision *const in, CoefficientPrecision beta, CoefficientPrecision *const out, CoefficientPrecision *work) {
    auto &input_partition           = trans == 'N' ? A.get_source_partition() : A.get_target_partition();
    auto &output_partition          = trans == 'N' ? A.get_target_partition() : A.get_source_partition();
    auto &global_to_local_operators = A.get_global_to_local_operators();
    auto &local_to_local_operators  = A.get_local_to_local_operators();

    int sizeWorld, rankWorld;
    auto comm = A.get_comm();
    MPI_Comm_rank(comm, &rankWorld);
    MPI_Comm_size(comm, &sizeWorld);
    int output_local_size   = output_partition.get_size_of_partition(rankWorld);
    int output_local_offset = output_partition.get_offset_of_partition(rankWorld);

    // offsets
    std::vector<int> recvcounts(sizeWorld);
    std::vector<int> displs(sizeWorld);
    displs[0] = 0;

    for (int i = 0; i < sizeWorld; i++) {
        recvcounts[i] = output_partition.get_size_of_partition(i);
        if (i > 0)
            displs[i] = displs[i - 1] + recvcounts[i - 1];
    }

    //
    const CoefficientPrecision *input = trans == 'N' ? in : in + input_partition.get_offset_of_partition(rankWorld);
    CoefficientPrecision *output, *output_buffer_ptr;
    std::vector<CoefficientPrecision> out_buffer;
    if (trans == 'N') {
        out_buffer.resize(work == nullptr ? output_local_size : 0);
        output = work == nullptr ? out_buffer.data() : work;
        if (beta != CoefficientPrecision(0)) {
            std::copy_n(out + displs[rankWorld], recvcounts[rankWorld], output);
        } else {
            std::fill_n(output, output_local_size, CoefficientPrecision(0));
        }
    } else if (trans != 'N' && beta != CoefficientPrecision(0)) {
        out_buffer.resize(work == nullptr ? output_partition.get_global_size() : 0);
        output_buffer_ptr = work == nullptr ? out_buffer.data() : work;
        std::copy_n(out, output_partition.get_global_size(), output_buffer_ptr);
        std::fill_n(out, output_partition.get_global_size(), CoefficientPrecision(0));
        output = out;
    } else {
        output = out;
    }

    // Product
    bool apply_beta = true;
    for (auto &local_operator : A.get_global_to_local_operators()) {
        local_operator->add_vector_product(trans, alpha, input, apply_beta ? beta : CoefficientPrecision(1), output);
        apply_beta = false;
    }

    for (auto &local_operator : A.get_local_to_local_operators()) {
        local_operator->add_vector_product(trans, alpha, in + input_partition.get_offset_of_partition(rankWorld), apply_beta ? beta : CoefficientPrecision(1), trans == 'N' ? output : output + output_local_offset);
        apply_beta = false;
    }

    // Communication
    if (trans == 'N') {
        MPI_Allgatherv(output, recvcounts[rankWorld], wrapper_mpi<CoefficientPrecision>::mpi_type(), out, &(recvcounts[0]), &(displs[0]), wrapper_mpi<CoefficientPrecision>::mpi_type(), comm);
    } else {
        MPI_Allreduce(MPI_IN_PLACE, output, output_partition.get_global_size(), wrapper_mpi<CoefficientPrecision>::mpi_type(), MPI_SUM, comm);
        if (beta != CoefficientPrecision(0)) {
            int n   = output_partition.get_global_size();
            int inc = 1;
            Blas<CoefficientPrecision>::axpy(&n, &beta, output_buffer_ptr, &inc, out, &inc);
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
/// @param work Optional buffer of size (M+N+m if trans=='N',  M+N+N if trans!='N' and beta!=0, M+N otherwise). Used if different of nullptr.
template <typename CoefficientPrecision>
void add_distributed_operator_vector_product_global_to_global(char trans, CoefficientPrecision alpha, const DistributedOperator<CoefficientPrecision> &A, const CoefficientPrecision *const in, CoefficientPrecision beta, CoefficientPrecision *const out, CoefficientPrecision *work) {
    auto &input_partition  = trans == 'N' ? A.get_source_partition() : A.get_target_partition();
    auto &output_partition = trans == 'N' ? A.get_target_partition() : A.get_source_partition();
    int ni                 = input_partition.get_global_size();
    int no                 = output_partition.get_global_size();
    std::vector<CoefficientPrecision> input_buffer(work == nullptr ? ni : 0);
    std::vector<CoefficientPrecision> output_buffer(work == nullptr ? no : 0);
    CoefficientPrecision *input_buffer_ptr  = work == nullptr ? input_buffer.data() : work;
    CoefficientPrecision *output_buffer_ptr = work == nullptr ? output_buffer.data() : work + ni;

    // Permutation
    input_partition.global_to_partition_numbering(in, input_buffer_ptr);
    if (beta != CoefficientPrecision(0)) {
        output_partition.global_to_partition_numbering(out, output_buffer_ptr);
    }

    // Product
    internal_add_distributed_operator_vector_product_global_to_global(trans, alpha, A, input_buffer_ptr, beta, output_buffer_ptr, work == nullptr ? nullptr : work + ni + no);

    // Permutation
    output_partition.partition_to_global_numbering(output_buffer_ptr, out);
}
} // namespace htool

#endif
