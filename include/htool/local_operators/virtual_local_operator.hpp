#ifndef HTOOL_VIRTUAL_LOCAL_OPERATOR_HPP
#define HTOOL_VIRTUAL_LOCAL_OPERATOR_HPP

namespace htool {
template <class T>
class VirtualLocalOperator {
  public:
    virtual void add_vector_product_global_to_local(T alpha, const T *const in, T beta, T *const out) const                = 0;
    virtual void add_matrix_product_global_to_local(T alpha, const T *const in, T beta, T *const out, int mu) const        = 0;
    virtual void add_vector_product_transp_local_to_global(T alpha, const T *const in, T beta, T *const out) const         = 0;
    virtual void add_matrix_product_transp_local_to_global(T alpha, const T *const in, T beta, T *const out, int mu) const = 0;

    virtual void sub_matrix_product_to_local(const T *const in, T *const out, int mu, int offset, int size) const = 0;

    virtual ~VirtualLocalOperator() {}

  protected:
    VirtualLocalOperator()                                            = default;
    VirtualLocalOperator(const VirtualLocalOperator &)                = default;
    VirtualLocalOperator(VirtualLocalOperator &&) noexcept            = default;
    VirtualLocalOperator &operator=(const VirtualLocalOperator &)     = default;
    VirtualLocalOperator &operator=(VirtualLocalOperator &&) noexcept = default;
};
} // namespace htool
#endif
