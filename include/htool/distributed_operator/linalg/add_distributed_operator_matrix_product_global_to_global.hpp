#ifndef HTOOL_DISTRIBUTED_OPERATOR_LINALG_ADD_DISTRIBUTED_OPERATOR_MATRIX_PRODUCT_GLOBAL_TO_GLOBAL_HPP
#define HTOOL_DISTRIBUTED_OPERATOR_LINALG_ADD_DISTRIBUTED_OPERATOR_MATRIX_PRODUCT_GLOBAL_TO_GLOBAL_HPP

#include "../distributed_operator.hpp"

namespace htool {

/// @brief
/// @tparam MatIn
/// @tparam MatOut
/// @tparam
/// @param trans
/// @param alpha
/// @param A
/// @param in
/// @param beta
/// @param out
/// @param work Optional buffer of size (N*mu+M*mu+m*mu if trans=='N', N*mu+M*mu+N*mu  if trans!='N' and beta!=0, N*mu+M*mu otherwise). Used if different of nullptr.
template <typename MatIn,
          typename MatOut,
          typename = std::enable_if_t<
              std::is_same<typename MatIn::value_type, typename MatOut::value_type>::value>>
void internal_add_distributed_operator_matrix_product_global_to_global(char trans, typename MatIn::value_type alpha, const DistributedOperator<typename MatIn::value_type> &A, const MatIn &in, typename MatIn::value_type beta, MatOut &out, typename MatIn::value_type *work) {
    using CoefficientPrecision = typename MatIn::value_type;
    int sizeWorld, rankWorld;
    auto comm = A.get_comm();
    MPI_Comm_rank(comm, &rankWorld);
    MPI_Comm_size(comm, &sizeWorld);
    auto &input_partition           = trans == 'N' ? A.get_source_partition() : A.get_target_partition();
    auto &output_partition          = trans == 'N' ? A.get_target_partition() : A.get_source_partition();
    int mu                          = in.nb_cols();
    int output_local_size           = output_partition.get_size_of_partition(rankWorld);
    int output_local_offset         = output_partition.get_offset_of_partition(rankWorld);
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

    // transpose
    Matrix<CoefficientPrecision> in_transposed;
    if (work == nullptr) {
        in_transposed.resize(in.nb_cols(), in.nb_rows());
    } else {
        in_transposed.assign(in.nb_cols(), in.nb_rows(), work, false);
    }
    transpose(in, in_transposed);
    Matrix<CoefficientPrecision> out_transposed;
    if (work == nullptr) {
        out_transposed.resize(out.nb_cols(), out.nb_rows());
    } else {
        out_transposed.assign(out.nb_cols(), out.nb_rows(), work + in.nb_cols() * in.nb_rows(), false);
    }

    //
    const CoefficientPrecision *input = trans == 'N' ? in_transposed.data() : in_transposed.data() + input_partition.get_offset_of_partition(rankWorld) * mu;
    CoefficientPrecision *output;
    Matrix<CoefficientPrecision> out_buffer;
    if (trans == 'N') {
        if (work == nullptr) {
            out_buffer.resize(mu, output_local_size);
        } else {
            out_buffer.assign(mu, output_local_size, work + in.nb_cols() * in.nb_rows() + out.nb_cols() * out.nb_rows(), false);
        }
        if (beta != CoefficientPrecision(0)) {
            for (int i = 0; i < output_local_size; i++) {
                for (int j = 0; j < mu; j++) {
                    out_buffer(j, i) = out(i + output_local_offset, j);
                }
            }
        } else {
            std::fill_n(out_buffer.data(), out_buffer.nb_cols() * out_buffer.nb_rows(), CoefficientPrecision(0));
        }
        output = out_buffer.data();
    } else if (trans != 'N' && beta != CoefficientPrecision(0)) {
        transpose(out, out_transposed);
        if (work == nullptr) {
            out_buffer.resize(out.nb_cols(), out.nb_rows());
        } else {
            out_buffer.assign(out.nb_cols(), out.nb_rows(), work + in.nb_cols() * in.nb_rows() + out.nb_cols() * out.nb_rows(), false);
            std::fill_n(out_buffer.data(), out_buffer.nb_cols() * out_buffer.nb_rows(), CoefficientPrecision(0));
        }
        output = out_buffer.data();
    } else {
        output = out_transposed.data();
    }

    // Product
    bool apply_beta = true;
    for (auto &local_operator : A.get_global_to_local_operators()) {
        local_operator->add_matrix_product_row_major(trans, alpha, input, apply_beta ? beta : CoefficientPrecision(1), output, mu);
        apply_beta = false;
    }

    for (auto &local_operator : A.get_local_to_local_operators()) {
        local_operator->add_matrix_product_row_major(trans, alpha, in_transposed.data() + input_partition.get_offset_of_partition(rankWorld) * mu, apply_beta ? beta : CoefficientPrecision(1), trans == 'N' ? output : output + output_local_offset * mu, mu);
        apply_beta = false;
    }

    // Communication
    if (trans == 'N') {
        MPI_Allgatherv(output, recvcounts[rankWorld], wrapper_mpi<CoefficientPrecision>::mpi_type(), out_transposed.data(), &(recvcounts[0]), &(displs[0]), wrapper_mpi<CoefficientPrecision>::mpi_type(), comm);
        transpose(out_transposed, out);

    } else {
        MPI_Allreduce(MPI_IN_PLACE, output, output_partition.get_global_size() * mu, wrapper_mpi<CoefficientPrecision>::mpi_type(), MPI_SUM, comm);
        if (beta != CoefficientPrecision(0)) {
            int n   = output_partition.get_global_size() * mu;
            int inc = 1;
            Blas<CoefficientPrecision>::axpy(&n, &beta, out_transposed.data(), &inc, out_buffer.data(), &inc);
            transpose(out_buffer, out);
        } else {
            transpose(out_transposed, out);
        }
    }
}

/// @brief
/// @tparam MatIn
/// @tparam MatOut
/// @tparam
/// @param trans
/// @param alpha
/// @param A
/// @param in
/// @param beta
/// @param out
/// @param work Optional buffer of size ((N+M)*(mu+1)+m*mu if trans=='N' and beta!=0, (N*(mu+1)+M*mu+m*mu if trans=='N' and beta==0, M+m*mu+N*(2*mu+1) if trans!='N' and beta!=0, M+m*mu+2*N*mu otherwise). Used if different of nullptr.
template <typename MatIn,
          typename MatOut,
          typename = std::enable_if_t<
              std::is_same<typename MatIn::value_type, typename MatOut::value_type>::value>>
void add_distributed_operator_matrix_product_global_to_global(char trans, typename MatIn::value_type alpha, const DistributedOperator<typename MatIn::value_type> &A, const MatIn &in, typename MatIn::value_type beta, MatOut &out, typename MatIn::value_type *work) {
    using CoefficientPrecision = typename MatIn::value_type;
    int sizeWorld, rankWorld;
    auto comm = A.get_comm();
    MPI_Comm_rank(comm, &rankWorld);
    MPI_Comm_size(comm, &sizeWorld);

    auto &input_partition  = trans == 'N' ? A.get_source_partition() : A.get_target_partition();
    auto &output_partition = trans == 'N' ? A.get_target_partition() : A.get_source_partition();
    int ni                 = input_partition.get_global_size();
    std::vector<CoefficientPrecision> input_buffer_transpose(work == nullptr ? ni : 0);
    CoefficientPrecision *input_buffer_transpose_ptr = work == nullptr ? input_buffer_transpose.data() : work;
    std::vector<CoefficientPrecision> output_buffer_transpose;
    CoefficientPrecision *output_buffer_transpose_ptr;
    int mu                          = in.nb_cols();
    auto &global_to_local_operators = A.get_global_to_local_operators();
    auto &local_to_local_operators  = A.get_local_to_local_operators();

    int input_local_size    = input_partition.get_size_of_partition(rankWorld);
    int input_local_offset  = input_partition.get_offset_of_partition(rankWorld);
    int output_local_size   = output_partition.get_size_of_partition(rankWorld);
    int output_local_offset = output_partition.get_offset_of_partition(rankWorld);

    // Offsets
    std::vector<int> recvcounts(sizeWorld);
    std::vector<int> displs(sizeWorld);
    displs[0] = 0;
    for (int i = 0; i < sizeWorld; i++) {
        recvcounts[i] = output_partition.get_size_of_partition(i) * mu;
        if (i > 0)
            displs[i] = displs[i - 1] + recvcounts[i - 1];
    }

    // transpose
    Matrix<CoefficientPrecision> in_transposed;
    if (work == nullptr) {
        in_transposed.resize(in.nb_cols(), trans == 'N' ? in.nb_rows() : input_local_size);
    } else {
        in_transposed.assign(in.nb_cols(), trans == 'N' ? in.nb_rows() : input_local_size, work + ni, false);
    }
    for (int j = 0; j < in.nb_cols(); j++) {
        input_partition.global_to_partition_numbering(in.data() + j * in.nb_rows(), input_buffer_transpose_ptr);
        if (trans == 'N') {
            for (int i = 0; i < in.nb_rows(); i++) {
                in_transposed(j, i) = input_buffer_transpose_ptr[i];
            }
        } else {
            for (int i = 0; i < input_local_size; i++) {
                in_transposed(j, i) = input_buffer_transpose_ptr[i + input_local_offset];
            }
        }
    }
    Matrix<CoefficientPrecision> out_transposed;
    if (work == nullptr) {
        out_transposed.resize(out.nb_cols(), out.nb_rows());
    } else {
        out_transposed.assign(out.nb_cols(), out.nb_rows(), trans == 'N' ? work + ni + in.nb_cols() * in.nb_rows() : work + ni + in.nb_cols() * input_local_size, false);
    }

    //
    const CoefficientPrecision *input = in_transposed.data();
    Matrix<CoefficientPrecision> out_buffer;
    if (trans == 'N') {
        if (work == nullptr) {
            out_buffer.resize(mu, output_local_size);
        } else {
            out_buffer.assign(mu, output_local_size, work + ni + in.nb_cols() * in.nb_rows() + out.nb_cols() * out.nb_rows(), false);
        }
        if (beta != CoefficientPrecision(0)) {
            if (work == nullptr)
                output_buffer_transpose.resize(out.nb_rows());
            output_buffer_transpose_ptr = work == nullptr ? output_buffer_transpose.data() : work + ni + in.nb_cols() * in.nb_rows() + out.nb_cols() * out.nb_rows() + mu * output_local_size;

            for (int j = 0; j < mu; j++) {
                output_partition.global_to_partition_numbering(out.data() + j * out.nb_rows(), output_buffer_transpose_ptr);
                for (int i = 0; i < output_local_size; i++) {
                    out_buffer(j, i) = output_buffer_transpose_ptr[i + output_local_offset];
                }
            }
        } else {
            std::fill_n(out_buffer.data(), out_buffer.nb_cols() * out_buffer.nb_rows(), CoefficientPrecision(0));
        }
    } else if (trans != 'N' && beta != CoefficientPrecision(0)) {
        if (work == nullptr)
            output_buffer_transpose.resize(out.nb_rows());
        output_buffer_transpose_ptr = work == nullptr ? output_buffer_transpose.data() : work + ni + in.nb_cols() * input_local_size + out.nb_cols() * out.nb_rows();
        for (int j = 0; j < out.nb_cols(); j++) {
            output_partition.global_to_partition_numbering(out.data() + j * out.nb_rows(), output_buffer_transpose_ptr);
            for (int i = 0; i < out.nb_rows(); i++) {
                out_transposed(j, i) = output_buffer_transpose_ptr[i];
            }
        }
        if (work == nullptr) {
            out_buffer.resize(out.nb_cols(), out.nb_rows());
        } else {
            out_buffer.assign(out.nb_cols(), out.nb_rows(), work + ni + in.nb_cols() * input_local_size + out.nb_cols() * out.nb_rows() + out.nb_rows(), false);
        }
    } else {
        if (work == nullptr) {
            out_buffer.resize(out.nb_cols(), out.nb_rows());
        } else {
            out_buffer.assign(out.nb_cols(), out.nb_rows(), work + ni + in.nb_cols() * input_local_size + out.nb_cols() * out.nb_rows(), false);
        }
    }

    // Product
    bool apply_beta = true;
    for (auto &local_operator : A.get_global_to_local_operators()) {
        local_operator->add_matrix_product_row_major(trans, alpha, input, apply_beta ? beta : CoefficientPrecision(1), out_buffer.data(), mu);
        apply_beta = false;
    }

    for (auto &local_operator : A.get_local_to_local_operators()) {
        local_operator->add_matrix_product_row_major(trans, alpha, trans == 'N' ? input + input_local_offset * mu : input, apply_beta ? beta : CoefficientPrecision(1), trans == 'N' ? out_buffer.data() : out_buffer.data() + output_local_offset * mu, mu);
        apply_beta = false;
    }

    // Communication
    Matrix<CoefficientPrecision> tmp_transposed, tmp;
    if (trans == 'N') {
        tmp_transposed.assign(out.nb_cols(), out.nb_rows(), out.data(), false);
        tmp.assign(out.nb_rows(), out.nb_cols(), out_transposed.data(), false);
        MPI_Allgatherv(out_buffer.data(), recvcounts[rankWorld], wrapper_mpi<CoefficientPrecision>::mpi_type(), tmp_transposed.data(), &(recvcounts[0]), &(displs[0]), wrapper_mpi<CoefficientPrecision>::mpi_type(), comm);
        transpose(tmp_transposed, tmp);
        for (int j = 0; j < out.nb_cols(); j++) {
            output_partition.partition_to_global_numbering(tmp.data() + j * tmp.nb_rows(), out.data() + out.nb_rows() * j);
        }

    } else {
        tmp_transposed.assign(out.nb_cols(), out.nb_rows(), out.data(), false);
        tmp.assign(out.nb_rows(), out.nb_cols(), out_transposed.data(), false);
        MPI_Allreduce(MPI_IN_PLACE, out_buffer.data(), output_partition.get_global_size() * mu, wrapper_mpi<CoefficientPrecision>::mpi_type(), MPI_SUM, comm);
        if (beta != CoefficientPrecision(0)) {
            int n   = output_partition.get_global_size() * mu;
            int inc = 1;
            Blas<CoefficientPrecision>::axpy(&n, &beta, out_transposed.data(), &inc, out_buffer.data(), &inc);
            transpose(out_buffer, tmp);
        } else {
            transpose(out_buffer, tmp);
        }
        for (int j = 0; j < out.nb_cols(); j++) {
            output_partition.partition_to_global_numbering(tmp.data() + j * tmp.nb_rows(), out.data() + out.nb_rows() * j);
        }
    }
}
} // namespace htool

#endif
