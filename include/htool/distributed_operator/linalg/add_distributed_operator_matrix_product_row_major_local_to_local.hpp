#ifndef HTOOL_DISTRIBUTED_OPERATOR_LINALG_ADD_DISTRIBUTED_OPERATOR_MATRIX_PRODUCT_ROW_MAJOR_LOCAL_TO_LOCAL_HPP
#define HTOOL_DISTRIBUTED_OPERATOR_LINALG_ADD_DISTRIBUTED_OPERATOR_MATRIX_PRODUCT_ROW_MAJOR_LOCAL_TO_LOCAL_HPP

#include "../distributed_operator.hpp"
#include "utility.hpp"

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
/// @param mu
/// @param work Optional buffer of size (N*mu if trans=='N', (N+n*sizeWorld)*mu otherwise). Used if different of nullptr.
template <typename MatIn,
          typename MatOut,
          typename = std::enable_if_t<
              std::is_same<typename MatIn::value_type, typename MatOut::value_type>::value>>
void internal_add_distributed_operator_matrix_product_row_major_local_to_local(char trans, typename MatIn::value_type alpha, const DistributedOperator<typename MatIn::value_type> &A, const MatIn &in, typename MatIn::value_type beta, MatOut &out, typename MatIn::value_type *work) {
    using CoefficientPrecision      = typename MatIn::value_type;
    auto &input_partition           = trans == 'N' ? A.get_source_partition() : A.get_target_partition();
    auto &output_partition          = trans == 'N' ? A.get_target_partition() : A.get_source_partition();
    int mu                          = in.nb_rows();
    auto &global_to_local_operators = A.get_global_to_local_operators();
    auto &local_to_local_operators  = A.get_local_to_local_operators();
    bool apply_beta                 = true;

    if (local_to_local_operators.size() > 0) {
        for (auto &local_operator : local_to_local_operators) {
            local_operator->add_matrix_product_row_major(trans, alpha, in.data(), apply_beta ? beta : CoefficientPrecision(1), out.data(), mu);
            apply_beta = false;
        }
    }

    std::vector<CoefficientPrecision> buffer;
    const CoefficientPrecision *input_ptr;
    CoefficientPrecision *output_ptr;
    int sizeWorld, rankWorld;
    if (trans == 'N') {
        buffer.resize(work == nullptr ? input_partition.get_global_size() * mu : 0);
        input_ptr  = (work == nullptr) ? buffer.data() : work;
        output_ptr = out.data();

        // Local to global
        local_to_global(input_partition, in.data(), (work == nullptr) ? buffer.data() : work, mu, A.get_comm());
    } else {
        MPI_Comm_rank(A.get_comm(), &rankWorld);
        MPI_Comm_size(A.get_comm(), &sizeWorld);
        buffer.resize(work == nullptr ? (output_partition.get_global_size() + output_partition.get_size_of_partition(rankWorld) * sizeWorld) * mu : 0);
        input_ptr  = in.data();
        output_ptr = (work == nullptr) ? buffer.data() : work;
        std::fill_n(output_ptr, (output_partition.get_global_size() + output_partition.get_size_of_partition(rankWorld) * sizeWorld) * mu, CoefficientPrecision(0));
        apply_beta = false;
    }

    // Product
    for (auto &local_operator : A.get_global_to_local_operators()) {
        local_operator->add_matrix_product_row_major(trans, alpha, input_ptr, apply_beta ? beta : CoefficientPrecision(1), output_ptr, mu);
        apply_beta = false;
    }

    if (trans != 'N') {
        std::vector<int> scounts(sizeWorld), rcounts(sizeWorld);
        std::vector<int> sdispls(sizeWorld), rdispls(sizeWorld);
        CoefficientPrecision *rbuf = output_ptr + output_partition.get_global_size() * mu;

        sdispls[0] = 0;
        rdispls[0] = 0;

        for (int i = 0; i < sizeWorld; i++) {
            scounts[i] = output_partition.get_size_of_partition(i) * mu;
            rcounts[i] = output_partition.get_size_of_partition(rankWorld) * mu;
            if (i > 0) {
                sdispls[i] = sdispls[i - 1] + scounts[i - 1];
                rdispls[i] = rdispls[i - 1] + rcounts[i - 1];
            }
        }

        MPI_Alltoallv(output_ptr, &(scounts[0]), &(sdispls[0]), wrapper_mpi<CoefficientPrecision>::mpi_type(), rbuf, &(rcounts[0]), &(rdispls[0]), wrapper_mpi<CoefficientPrecision>::mpi_type(), A.get_comm());

        int inc                  = 1;
        CoefficientPrecision one = 1;
        int local_size           = out.nb_rows() * out.nb_cols();
        if (local_to_local_operators.size() == 0)
            Blas<CoefficientPrecision>::scal(&local_size, &beta, out.data(), &inc);
        for (int i = 0; i < sizeWorld; i++) {
            Blas<CoefficientPrecision>::axpy(&local_size, &one, rbuf + rdispls[i], &inc, out.data(), &inc);
        }
    }
}

} // namespace htool

#endif
