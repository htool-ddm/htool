#ifndef HTOOL_DISTRIBUTED_OPERATOR_LINALG_ADD_DISTRIBUTED_OPERATOR_MATRIX_PRODUCT_LOCAL_TO_LOCAL_HPP
#define HTOOL_DISTRIBUTED_OPERATOR_LINALG_ADD_DISTRIBUTED_OPERATOR_MATRIX_PRODUCT_LOCAL_TO_LOCAL_HPP

#include "../distributed_operator.hpp"
#include "add_distributed_operator_matrix_product_row_major_local_to_local.hpp"

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
/// @param work Optional buffer of size ((m+n+N)*mu if trans=='N', (m+n+N+n*sizeWorld)*mu otherwise). Used if different of nullptr.
template <typename MatIn,
          typename MatOut,
          typename = std::enable_if_t<
              std::is_same<typename MatIn::value_type, typename MatOut::value_type>::value>>
void internal_add_distributed_operator_matrix_product_local_to_local(char trans, typename MatIn::value_type alpha, const DistributedOperator<typename MatIn::value_type> &A, const MatIn &in, typename MatIn::value_type beta, MatOut &out, typename MatIn::value_type *work) {
    using CoefficientPrecision = typename MatIn::value_type;

    // transpose
    Matrix<CoefficientPrecision> in_transposed;
    Matrix<CoefficientPrecision> out_transposed;
    if (work == nullptr) {
        in_transposed.resize(in.nb_cols(), in.nb_rows());
        out_transposed.resize(out.nb_cols(), out.nb_rows());
    } else {
        in_transposed.assign(in.nb_cols(), in.nb_rows(), work, false);
        out_transposed.assign(out.nb_cols(), out.nb_rows(), work + in.nb_cols() * in.nb_rows(), false);
    }

    transpose(in, in_transposed);
    if (beta != CoefficientPrecision(0)) {
        transpose(out, out_transposed);
    }

    // Product
    internal_add_distributed_operator_matrix_product_row_major_local_to_local(trans, alpha, A, in_transposed, beta, out_transposed, work == nullptr ? nullptr : work + in.nb_cols() * in.nb_rows() + out.nb_cols() * out.nb_rows());

    // transpose
    transpose(out_transposed, out);
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
/// @param work Optional buffer of size ( m*(mu+1)+n*(mu+1)+N*mu if trans=='N',  m*(mu+1)+n*(mu+1)+N+n*sizeWorld otherwise). Used if different of nullptr.
template <typename MatIn,
          typename MatOut,
          typename = std::enable_if_t<
              std::is_same<typename MatIn::value_type, typename MatOut::value_type>::value>>
void add_distributed_operator_matrix_product_local_to_local(char trans, typename MatIn::value_type alpha, const DistributedOperator<typename MatIn::value_type> &A, const MatIn &in, typename MatIn::value_type beta, MatOut &out, typename MatIn::value_type *work) {
    using CoefficientPrecision = typename MatIn::value_type;
    int sizeWorld, rankWorld;
    auto comm = A.get_comm();
    MPI_Comm_rank(comm, &rankWorld);
    MPI_Comm_size(comm, &sizeWorld);
    auto &input_partition  = trans == 'N' ? A.get_source_partition() : A.get_target_partition();
    auto &output_partition = trans == 'N' ? A.get_target_partition() : A.get_source_partition();

    int ni = input_partition.get_size_of_partition(rankWorld);
    int no = output_partition.get_size_of_partition(rankWorld);

    // transpose + permutation
    Matrix<CoefficientPrecision> in_transposed;
    Matrix<CoefficientPrecision> out_transposed;
    if (work == nullptr) {
        in_transposed.resize(in.nb_cols(), in.nb_rows());
        out_transposed.resize(out.nb_cols(), out.nb_rows());
    } else {
        in_transposed.assign(in.nb_cols(), in.nb_rows(), work, false);
        out_transposed.assign(out.nb_cols(), out.nb_rows(), work + in.nb_cols() * in.nb_rows(), false);
    }
    std::vector<CoefficientPrecision> input_buffer_transpose(work == nullptr ? ni : 0);
    std::vector<CoefficientPrecision> output_buffer_transpose(work == nullptr ? no : 0);
    CoefficientPrecision *input_buffer_transpose_ptr  = work == nullptr ? input_buffer_transpose.data() : work + in.nb_cols() * in.nb_rows() + out.nb_cols() * out.nb_rows();
    CoefficientPrecision *output_buffer_transpose_ptr = work == nullptr ? output_buffer_transpose.data() : work + in.nb_cols() * in.nb_rows() + out.nb_cols() * out.nb_rows() + ni;

    for (int j = 0; j < in.nb_cols(); j++) {
        input_partition.local_to_local_partition_numbering(rankWorld, in.data() + j * in.nb_rows(), input_buffer_transpose_ptr);
        for (int i = 0; i < in.nb_rows(); i++) {
            in_transposed(j, i) = input_buffer_transpose_ptr[i];
        }
    }

    if (beta != CoefficientPrecision(0)) {
        for (int j = 0; j < out.nb_cols(); j++) {
            output_partition.local_to_local_partition_numbering(rankWorld, out.data() + j * out.nb_rows(), output_buffer_transpose_ptr);
            for (int i = 0; i < out.nb_rows(); i++) {
                out_transposed(j, i) = output_buffer_transpose_ptr[i];
            }
        }
    }

    // Product
    internal_add_distributed_operator_matrix_product_row_major_local_to_local(trans, alpha, A, in_transposed, beta, out_transposed, work == nullptr ? nullptr : work + in.nb_cols() * in.nb_rows() + out.nb_cols() * out.nb_rows() + ni + no);

    // transpose + permutation
    for (int j = 0; j < out.nb_cols(); j++) {
        for (int i = 0; i < out.nb_rows(); i++) {
            output_buffer_transpose_ptr[i] = out_transposed(j, i);
        }
        output_partition.local_partition_to_local_numbering(rankWorld, output_buffer_transpose_ptr, out.data() + j * out.nb_rows());
    }
}
} // namespace htool

#endif
