#ifndef HTOOL_DISTRIBUTED_OPERATOR_VIRTUAL_GLOBAL_TO_LOCAL_OPERATOR_HPP
#define HTOOL_DISTRIBUTED_OPERATOR_VIRTUAL_GLOBAL_TO_LOCAL_OPERATOR_HPP

namespace htool {
/// @brief Interface representing
/// @tparam T
template <class T>
class VirtualGlobalToLocalOperator {
  public:
    /// @brief
    /// @param trans
    /// @param alpha
    /// @param in Global input vector when trans=='N'. Local input vector when trans!=N.
    /// @param beta
    /// @param out Local output vector when trans=='N'. Global output vector when trans=='N'.
    virtual void add_vector_product(char trans, T alpha, const T *const in, T beta, T *const out) const = 0;

    /// @brief
    /// @param trans
    /// @param alpha
    /// @param in Global input row-major matrix when trans=='N'. Local input row-major matrix when trans!=N.
    /// @param beta
    /// @param out Local input row-major matrix when trans=='N'. Global input row-major matrix when trans!=N.
    /// @param mu Number of columns for in and out.
    virtual void add_matrix_product_row_major(char trans, T alpha, const T *const in, T beta, T *const out, int mu) const = 0;

    /// @brief
    /// @param in
    /// @param out
    /// @param mu
    /// @param offset
    /// @param size
    virtual void add_sub_matrix_product_to_local(const T *const in, T *const out, int mu, int offset, int size) const = 0;

    virtual ~VirtualGlobalToLocalOperator() {}

  protected:
    VirtualGlobalToLocalOperator()                                                    = default;
    VirtualGlobalToLocalOperator(const VirtualGlobalToLocalOperator &)                = default;
    VirtualGlobalToLocalOperator(VirtualGlobalToLocalOperator &&) noexcept            = default;
    VirtualGlobalToLocalOperator &operator=(const VirtualGlobalToLocalOperator &)     = default;
    VirtualGlobalToLocalOperator &operator=(VirtualGlobalToLocalOperator &&) noexcept = default;
};
} // namespace htool
#endif
