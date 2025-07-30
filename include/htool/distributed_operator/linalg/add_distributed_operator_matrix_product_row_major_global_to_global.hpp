#ifndef HTOOL_DISTRIBUTED_OPERATOR_LINALG_ADD_DISTRIBUTED_OPERATOR_MATRIX_PRODUCT_ROW_MAJOR_GLOBAL_TO_GLOBAL_HPP
#define HTOOL_DISTRIBUTED_OPERATOR_LINALG_ADD_DISTRIBUTED_OPERATOR_MATRIX_PRODUCT_ROW_MAJOR_GLOBAL_TO_GLOBAL_HPP

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
/// @param work Optional buffer of size (m*mu if trans=='N', N*mu if trans!='N' and beta!=0, 0 otherwise). Used if different of nullptr.
template <typename CoefficientPrecision>
void internal_add_distributed_operator_matrix_product_row_major_global_to_global(char trans, CoefficientPrecision alpha, const DistributedOperator<CoefficientPrecision> &A, const Matrix<CoefficientPrecision> &in, CoefficientPrecision beta, Matrix<CoefficientPrecision> &out, CoefficientPrecision *work) {
    int sizeWorld, rankWorld;
    auto comm = A.get_comm();
    MPI_Comm_rank(comm, &rankWorld);
    MPI_Comm_size(comm, &sizeWorld);
    auto &input_partition           = trans == 'N' ? A.get_source_partition() : A.get_target_partition();
    auto &output_partition          = trans == 'N' ? A.get_target_partition() : A.get_source_partition();
    int output_local_size           = output_partition.get_size_of_partition(rankWorld);
    int output_local_offset         = output_partition.get_offset_of_partition(rankWorld);
    int mu                          = in.nb_rows();
    auto &global_to_local_operators = A.get_global_to_local_operators();
    auto &local_to_local_operators  = A.get_local_to_local_operators();

    // Offsets
    std::vector<int> recvcounts(sizeWorld);
    std::vector<int> displs(sizeWorld);
    displs[0] = 0;
    for (int i = 0; i < sizeWorld; i++) {
        recvcounts[i] = output_partition.get_size_of_partition(i) * mu;
        if (i > 0)
            displs[i] = displs[i - 1] + recvcounts[i - 1];
    }

    //
    const CoefficientPrecision *input = trans == 'N' ? in.data() : in.data() + input_partition.get_offset_of_partition(rankWorld) * mu;
    CoefficientPrecision *output, *output_buffer_ptr;
    std::vector<CoefficientPrecision> out_buffer;
    if (trans == 'N') {
        out_buffer.resize(work == nullptr ? output_local_size * mu : 0);
        output = work == nullptr ? out_buffer.data() : work;
        if (beta != CoefficientPrecision(0)) {
            std::copy_n(out.data() + displs[rankWorld], recvcounts[rankWorld], output);
        } else {
            std::fill_n(output, output_local_size * mu, CoefficientPrecision(0));
        }
    } else if (trans != 'N' && beta != CoefficientPrecision(0)) {
        out_buffer.resize(work == nullptr ? output_partition.get_global_size() * mu : 0);
        output_buffer_ptr = work == nullptr ? out_buffer.data() : work;
        std::copy_n(out.data(), output_partition.get_global_size() * mu, output_buffer_ptr);
        std::fill_n(out.data(), output_partition.get_global_size() * mu, CoefficientPrecision(0));
        output = out.data();
    } else {
        output = out.data();
    }

    // Product
    bool apply_beta = true;
    for (auto &local_operator : A.get_global_to_local_operators()) {
        local_operator->add_matrix_product_row_major(trans, alpha, input, apply_beta ? beta : CoefficientPrecision(1), output, mu);
        apply_beta = false;
    }

    for (auto &local_operator : A.get_local_to_local_operators()) {
        local_operator->add_matrix_product_row_major(trans, alpha, in.data() + input_partition.get_offset_of_partition(rankWorld) * mu, apply_beta ? beta : CoefficientPrecision(1), trans == 'N' ? output : output + output_local_offset * mu, mu);
        apply_beta = false;
    }

    // Communication
    if (trans == 'N') {
        MPI_Allgatherv(output, recvcounts[rankWorld], wrapper_mpi<CoefficientPrecision>::mpi_type(), out.data(), &(recvcounts[0]), &(displs[0]), wrapper_mpi<CoefficientPrecision>::mpi_type(), comm);
    } else {
        MPI_Allreduce(MPI_IN_PLACE, output, output_partition.get_global_size() * mu, wrapper_mpi<CoefficientPrecision>::mpi_type(), MPI_SUM, comm);
        if (beta != CoefficientPrecision(0)) {
            int n   = output_partition.get_global_size() * mu;
            int inc = 1;
            Blas<CoefficientPrecision>::axpy(&n, &beta, output_buffer_ptr, &inc, out.data(), &inc);
        }
    }
}

} // namespace htool

#endif
