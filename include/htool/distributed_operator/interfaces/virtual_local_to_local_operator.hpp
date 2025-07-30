#ifndef HTOOL_DISTRIBUTED_OPERATOR_VIRTUAL_LOCAL_TO_LOCAL_OPERATOR_HPP
#define HTOOL_DISTRIBUTED_OPERATOR_VIRTUAL_LOCAL_TO_LOCAL_OPERATOR_HPP

namespace htool {
/// @brief Interface representing
/// @tparam T
template <class T>
class VirtualLocalToLocalOperator {
  public:
    /// @brief
    /// @param trans
    /// @param alpha
    /// @param in Local input vector.
    /// @param beta
    /// @param out Local output vector.
    virtual void add_vector_product(char trans, T alpha, const T *const in, T beta, T *const out) const = 0;

    /// @brief
    /// @param trans
    /// @param alpha
    /// @param in Local input row-major matrix.
    /// @param beta
    /// @param out Local output row-major matrix.
    /// @param mu Number of columns for in and out.
    virtual void add_matrix_product_row_major(char trans, T alpha, const T *const in, T beta, T *const out, int mu) const = 0;

    /// @brief
    /// @param in
    /// @param out
    /// @param mu
    /// @param offset
    /// @param size
    virtual void add_sub_matrix_product_to_local(const T *const in, T *const out, int mu, int offset, int size) const = 0;

    virtual ~VirtualLocalToLocalOperator() {}

  protected:
    VirtualLocalToLocalOperator()                                                   = default;
    VirtualLocalToLocalOperator(const VirtualLocalToLocalOperator &)                = default;
    VirtualLocalToLocalOperator(VirtualLocalToLocalOperator &&) noexcept            = default;
    VirtualLocalToLocalOperator &operator=(const VirtualLocalToLocalOperator &)     = default;
    VirtualLocalToLocalOperator &operator=(VirtualLocalToLocalOperator &&) noexcept = default;
};
} // namespace htool
#endif
