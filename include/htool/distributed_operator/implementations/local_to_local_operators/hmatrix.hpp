
#ifndef HTOOL_DISTRIBUTED_OPERATOR_LOCAL_TO_LOCAL_HMATRIX_HPP
#define HTOOL_DISTRIBUTED_OPERATOR_LOCAL_TO_LOCAL_HMATRIX_HPP

#include "../../../clustering/cluster_node.hpp" // for Cluster
#include "../../../hmatrix/hmatrix.hpp"
#include "../../../hmatrix/linalg/add_hmatrix_matrix_product_row_major.hpp"
#include "../../../hmatrix/linalg/add_hmatrix_vector_product.hpp"
#include "../../../misc/misc.hpp"
#include "../../interfaces/virtual_local_to_local_operator.hpp"

namespace htool {

template <typename CoefficientPrecision, typename CoordinatePrecision = underlying_type<CoefficientPrecision>>
class LocalToLocalHMatrix final : public VirtualLocalToLocalOperator<CoefficientPrecision> {
    const HMatrix<CoefficientPrecision, CoordinatePrecision> &m_data;

  public:
    LocalToLocalHMatrix(const HMatrix<CoefficientPrecision, CoordinatePrecision> &hmatrix) : m_data(hmatrix) {}

    LocalToLocalHMatrix(const LocalToLocalHMatrix &)                                   = default;
    LocalToLocalHMatrix &operator=(const LocalToLocalHMatrix &)                        = default;
    LocalToLocalHMatrix(LocalToLocalHMatrix &&LocalToLocalHMatrix) noexcept            = default;
    LocalToLocalHMatrix &operator=(LocalToLocalHMatrix &&LocalToLocalHMatrix) noexcept = default;
    ~LocalToLocalHMatrix()                                                             = default;

    void add_vector_product(char trans, CoefficientPrecision alpha, const CoefficientPrecision *in, CoefficientPrecision beta, CoefficientPrecision *out) const override {
        openmp_internal_add_hmatrix_vector_product(trans, alpha, m_data, in, beta, out);
    }
    void add_matrix_product_row_major(char trans, CoefficientPrecision alpha, const CoefficientPrecision *in, CoefficientPrecision beta, CoefficientPrecision *out, int mu) const override {
        openmp_internal_add_hmatrix_matrix_product_row_major(trans, 'N', alpha, m_data, in, beta, out, mu);
    }

    void add_sub_matrix_product_to_local(const CoefficientPrecision *const in, CoefficientPrecision *const out, int mu, int offset, int size) const override {
        int source_offset   = m_data.get_source_cluster().get_offset();
        int source_size     = m_data.get_source_cluster().get_size();
        int source_end      = source_size + source_offset;
        int end             = size + offset;
        int temp_offset     = std::max(offset, source_offset);
        int temp_end        = std::min(source_end, end);
        bool is_output_null = temp_end - temp_offset <= 0 ? true : false;
        if (offset == source_offset && temp_end == source_end) {
            add_matrix_product_row_major('N', 1, in, 1, out, mu);
        } else {
            const CoefficientPrecision *const temp_in = in + temp_offset - offset;
            int temp_size                             = temp_end - temp_offset;
            std::vector<CoefficientPrecision> extension_by_zero(source_size * mu, 0);
            if (!is_output_null) {
                std::copy_n(temp_in, temp_size * mu, extension_by_zero.data() + (offset - source_offset) * mu);
            }
            add_matrix_product_row_major('N', 1, extension_by_zero.data(), 1, out, mu);
        }
    }

    const HMatrix<CoefficientPrecision, CoordinatePrecision> &get_hmatrix() const { return *m_data.get(); }
};
} // namespace htool
#endif
