#ifndef HTOOL_VIRTUAL_LOCAL_OPERATOR_HPP
#define HTOOL_VIRTUAL_LOCAL_OPERATOR_HPP

#include "virtual_cluster.hpp"

namespace htool {
template <class T>
class VirtualLocalOperator {
  public:
    // // -- Getters --
    // virtual const VirtualCluster *cluster_tree_target() const = 0;
    // virtual const VirtualCluster *cluster_tree_source() const = 0;

    // -- Operations --
    virtual void add_vector_product_global_to_local(T alpha, int size_in, const T *const in, T beta, int size_out, T *const out) const = 0;

    virtual void add_matrix_product_global_to_local(T alpha, int size_in, const T *const in, T beta, int size_out, T *const out, int mu) const = 0;

    virtual void add_vector_product_transp_local_to_global(T alpha, int size_in, const T *const in, T beta, int size_out, T *const out) const = 0;

    virtual void add_matrix_product_transp_local_to_global(T alpha, int size_in, const T *const in, T beta, int size_out, T *const out, int mu) const = 0;

    //  -- Destructors --
    virtual ~VirtualLocalOperator(){};

  protected:
    //  -- Constructors --
    VirtualLocalOperator()                             = default;
    VirtualLocalOperator(const VirtualLocalOperator &) = default;
    VirtualLocalOperator(VirtualLocalOperator &&)      = default;

    //  -- Assignment --
    VirtualLocalOperator &operator=(const VirtualLocalOperator &) = default;
    VirtualLocalOperator &operator=(VirtualLocalOperator &&)      = default;
};
} // namespace htool
#endif
