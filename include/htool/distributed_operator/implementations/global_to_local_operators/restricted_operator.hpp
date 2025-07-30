
#ifndef HTOOL_DISTRIBUTED_OPERATOR_LOCAL_OPERATOR_HPP
#define HTOOL_DISTRIBUTED_OPERATOR_LOCAL_OPERATOR_HPP

#include "../../../matrix/matrix.hpp"                            // for Matrix...
#include "../../../misc/logger.hpp"                              // for Logger, Log...
#include "../../../misc/misc.hpp"                                // for conj_if_com...
#include "../../interfaces/virtual_global_to_local_operator.hpp" // for VirtualLoca...
#include "../../local_renumbering.hpp"                           // for VirtualLocal...
#include <algorithm>                                             // for max, copy_n
#include <string>                                                // for basic_string
#include <vector>                                                // for vector

namespace htool {

template <typename CoefficientPrecision>
class RestrictedGlobalToLocalOperator : public VirtualGlobalToLocalOperator<CoefficientPrecision> {
  protected:
    LocalRenumbering m_local_target_renumbering;
    LocalRenumbering m_local_source_renumbering;

    bool m_target_use_permutation_to_mvprod{false}; // Permutation used when add_mvprod, useful for offdiag
    bool m_source_use_permutation_to_mvprod{false}; // Permutation used when add_mvprod, useful for offdiag

    RestrictedGlobalToLocalOperator(LocalRenumbering local_target_renumbering, LocalRenumbering local_source_renumbering, bool target_use_permutation_to_mvprod = false, bool source_use_permutation_to_mvprod = false) : m_local_target_renumbering(local_target_renumbering), m_local_source_renumbering(local_source_renumbering), m_target_use_permutation_to_mvprod(target_use_permutation_to_mvprod), m_source_use_permutation_to_mvprod(source_use_permutation_to_mvprod) {}

    RestrictedGlobalToLocalOperator(const RestrictedGlobalToLocalOperator &)                             = default;
    RestrictedGlobalToLocalOperator &operator=(const RestrictedGlobalToLocalOperator &)                  = default;
    RestrictedGlobalToLocalOperator(RestrictedGlobalToLocalOperator &&LocalOperator) noexcept            = default;
    RestrictedGlobalToLocalOperator &operator=(RestrictedGlobalToLocalOperator &&LocalOperator) noexcept = default;

    /// @brief
    /// @param trans
    /// @param alpha
    /// @param in Local input vector when trans=='N'. Local input vector when trans!=N.
    /// @param beta
    /// @param out Local output vector when trans=='N'. Local output vector when trans=='N'.
    virtual void local_add_vector_product(char trans, CoefficientPrecision alpha, const CoefficientPrecision *in, CoefficientPrecision beta, CoefficientPrecision *out) const = 0;

    /// @brief
    /// @param trans
    /// @param alpha
    /// @param in ocal input row-major matrix when trans=='N'. Local input row-major matrix when trans!=N.
    /// @param beta
    /// @param out Local input row-major matrix when trans=='N'. Mpcam input row-major matrix when trans!=N.
    /// @param mu Number of columns for in and out.
    virtual void local_add_matrix_product_row_major(char trans, CoefficientPrecision alpha, const CoefficientPrecision *in, CoefficientPrecision beta, CoefficientPrecision *out, int mu) const = 0;

    virtual void local_add_sub_matrix_product_to_local(const CoefficientPrecision *const in, CoefficientPrecision *const out, int mu, int offset, int size) const {
        // std::vector<CoefficientPrecision> temp(m_local_source_renumbering.get_global_size() * mu, 0);
        // std::copy_n(in, size * mu, temp.data() + offset * mu);
        // add_matrix_product_row_major('N',1, temp.data(), 1, out, mu);
        std::vector<CoefficientPrecision> temp(m_local_source_renumbering.get_size() * mu, 0);
        std::copy_n(in, size * mu, temp.data() + (offset - m_local_source_renumbering.get_offset()) * mu);
        local_add_matrix_product_row_major('N', 1, temp.data(), 1, out, mu);
    }

  public:
    void add_vector_product(char trans, CoefficientPrecision alpha, const CoefficientPrecision *in, CoefficientPrecision beta, CoefficientPrecision *out) const override {

        auto input_use_permutation_to_mvprod  = trans == 'N' ? m_source_use_permutation_to_mvprod : m_target_use_permutation_to_mvprod;
        auto output_use_permutation_to_mvprod = trans == 'N' ? m_target_use_permutation_to_mvprod : m_source_use_permutation_to_mvprod;

        auto &input_renumbering  = trans == 'N' ? m_local_source_renumbering : m_local_target_renumbering;
        auto &output_renumbering = trans == 'N' ? m_local_target_renumbering : m_local_source_renumbering;

        if (trans != 'N' && output_renumbering.get_size() != output_renumbering.get_global_size()) {
            int global_size = output_renumbering.get_global_size();
            int inc         = 1;
            Blas<CoefficientPrecision>::scal(&global_size, &beta, out, &inc);
            beta = 1;
        }

        // Permutation
        std::vector<CoefficientPrecision> buffer_in(input_use_permutation_to_mvprod ? input_renumbering.get_size() : 0);
        std::vector<CoefficientPrecision> buffer_out(output_use_permutation_to_mvprod ? output_renumbering.get_size() : 0);

        if (input_use_permutation_to_mvprod) {
            user_to_htool_numbering(input_renumbering, trans == 'N' ? in + input_renumbering.get_offset() : in, buffer_in.data());
        }
        if (output_use_permutation_to_mvprod && beta != CoefficientPrecision(0)) {
            user_to_htool_numbering(output_renumbering, trans == 'N' ? out : out + output_renumbering.get_offset(), buffer_out.data());
        }

        const CoefficientPrecision *input;
        if (input_use_permutation_to_mvprod) {
            input = buffer_in.data();
        } else {
            input = trans == 'N' ? in + input_renumbering.get_offset() : in;
        }
        CoefficientPrecision *output;
        if (output_use_permutation_to_mvprod) {
            output = buffer_out.data();
        } else {
            output = trans == 'N' ? out : out + output_renumbering.get_offset();
        }

        local_add_vector_product(trans, alpha, input, beta, output);

        // Permutation
        if (output_use_permutation_to_mvprod) {
            htool_to_user_numbering(output_renumbering, output, trans == 'N' ? out : out + output_renumbering.get_offset());
        }
    }
    void add_matrix_product_row_major(char trans, CoefficientPrecision alpha, const CoefficientPrecision *in, CoefficientPrecision beta, CoefficientPrecision *out, int mu) const override {
        auto input_use_permutation_to_mvprod  = trans == 'N' ? m_source_use_permutation_to_mvprod : m_target_use_permutation_to_mvprod;
        auto output_use_permutation_to_mvprod = trans == 'N' ? m_target_use_permutation_to_mvprod : m_source_use_permutation_to_mvprod;

        auto &input_renumbering  = trans == 'N' ? m_local_source_renumbering : m_local_target_renumbering;
        auto &output_renumbering = trans == 'N' ? m_local_target_renumbering : m_local_source_renumbering;

        int ni = input_renumbering.get_size();
        int no = output_renumbering.get_size();

        if (trans != 'N' && output_renumbering.get_size() != output_renumbering.get_global_size()) {
            int global_size = output_renumbering.get_global_size() * mu;
            int inc         = 1;
            Blas<CoefficientPrecision>::scal(&global_size, &beta, out, &inc);
            beta = 1;
        }

        //
        std::vector<CoefficientPrecision> buffer_in(input_use_permutation_to_mvprod && trans == 'N' ? ni * mu : 0);

        std::vector<CoefficientPrecision> buffer_out(output_use_permutation_to_mvprod && trans != 'N' ? no * mu : 0);

        // Permutation + transpose
        if (input_use_permutation_to_mvprod && trans == 'N') {
            auto permutation = input_renumbering.get_permutation();
            for (int j = 0; j < ni; j++) {
                for (int i = 0; i < mu; i++) {
                    buffer_in[mu * j + i] = in[mu * permutation[j + input_renumbering.get_offset()] + i];
                }
            }
        } else if (input_use_permutation_to_mvprod && trans != 'N') {
            Logger::get_instance().log(LogLevel::CRITICAL, "Missing permutation."); // LCOV_EXCL_LINE
            // auto &permutation = input_renumbering.get_permutation();
            // for (int j = 0; j < ni; j++) {
            //     for (int i = 0; i < mu; i++) {
            //         buffer_in[mu * j + i] = in[mu * permutation[j + input_renumbering.get_offset()] + i];
            //     }
            // }
        }
        const CoefficientPrecision *input;
        if (input_use_permutation_to_mvprod) {
            input = buffer_in.data();
        } else {
            input = trans == 'N' ? in + input_renumbering.get_offset() * mu : in;
        }
        CoefficientPrecision *output;
        if (output_use_permutation_to_mvprod) {
            output = buffer_out.data();
        } else {
            output = trans == 'N' ? out : out + output_renumbering.get_offset() * mu;
        }

        // Local to local product
        local_add_matrix_product_row_major(trans, alpha, input, beta, output, mu);

        // Permutation TODO
        if (output_use_permutation_to_mvprod && trans == 'N') {
            Logger::get_instance().log(LogLevel::CRITICAL, "Missing permutation."); // LCOV_EXCL_LINE
        } else if (output_use_permutation_to_mvprod && trans != 'N') {
            auto permutation = output_renumbering.get_permutation();
            for (int j = 0; j < no; j++) {
                for (int i = 0; i < mu; i++) {
                    out[mu * permutation[j + output_renumbering.get_offset()] + i] = output[mu * j + i];
                }
            }
        }
    }

    void add_sub_matrix_product_to_local(const CoefficientPrecision *const in, CoefficientPrecision *const out, int mu, int offset, int size) const override {
        int source_offset   = m_local_source_renumbering.get_offset();
        int source_size     = m_local_source_renumbering.get_size();
        int source_end      = source_size + source_offset;
        int end             = size + offset;
        int temp_offset     = std::max(offset, source_offset);
        int temp_end        = std::min(source_end, end);
        bool is_output_null = temp_end - temp_offset <= 0 ? true : false;
        if (offset == source_offset && temp_end == source_end) {
            local_add_matrix_product_row_major('N', 1, in, 1, out, mu);
        } else {
            const CoefficientPrecision *const temp_in = in + temp_offset - offset;
            int temp_size                             = temp_end - temp_offset;
            if (is_output_null) {
                std::vector<CoefficientPrecision> temp(source_size * mu, 0);
                local_add_matrix_product_row_major('N', 1, temp.data(), 1, out, mu);

            } else {
                local_add_sub_matrix_product_to_local(temp_in, out, mu, temp_offset, temp_size);
            }
        }
    }

    virtual ~RestrictedGlobalToLocalOperator() = default;
};
} // namespace htool
#endif
