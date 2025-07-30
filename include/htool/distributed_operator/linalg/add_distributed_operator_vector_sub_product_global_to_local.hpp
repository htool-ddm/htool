#ifndef HTOOL_DISTRIBUTED_OPERATOR_LINALG_ADD_DISTRIBUTED_OPERATOR_VECTOR_SUB_PRODUCT_GLOBAL_TO_LOCAL_HPP
#define HTOOL_DISTRIBUTED_OPERATOR_LINALG_ADD_DISTRIBUTED_OPERATOR_VECTOR_SUB_PRODUCT_GLOBAL_TO_LOCAL_HPP

#include "../distributed_operator.hpp"
#include "utility.hpp"

namespace htool {

template <typename CoefficientPrecision>
void internal_add_distributed_operator_vector_sub_product_global_to_local(const DistributedOperator<CoefficientPrecision> &A, const CoefficientPrecision *const in, CoefficientPrecision *const out, int mu, int offset, int size) {

    // Product
    for (auto &local_operator : A.get_global_to_local_operators()) {
        local_operator->add_sub_matrix_product_to_local(in, out, mu, offset, size);
    }
    for (auto &local_operator : A.get_local_to_local_operators()) {
        local_operator->add_sub_matrix_product_to_local(in, out, mu, offset, size);
    }
}

} // namespace htool

#endif
