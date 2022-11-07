#ifndef HTOOL_VIRTUAL_DISTRIBUTED_OPERATOR_HPP
#define HTOOL_VIRTUAL_DISTRIBUTED_OPERATOR_HPP

#include <map>
#include <mpi.h>
#include <vector>

namespace htool {
template <class T>
class VirtualDistributedOperator {
  public:
    // // Matrix-vector products in user numbering
    // virtual void mvprod_global_to_global(const T *const in, T *const out, const int &mu = 1) const                  = 0;
    // virtual void mvprod_local_to_local(const T *const in, T *const out, const int &mu = 1, T *work = nullptr) const = 0;

    // virtual void mvprod_transp_global_to_global(const T *const in, T *const out, const int &mu = 1) const                  = 0;
    // virtual void mvprod_transp_local_to_local(const T *const in, T *const out, const int &mu = 1, T *work = nullptr) const = 0;

    // // Matrix-vector products in htool numbering
    // virtual void mymvprod_local_to_local(const T *const in, T *const out, const int &mu = 1, T *work = nullptr) const = 0;
    // virtual void mymvprod_global_to_local(const T *const in, T *const out, const int &mu = 1) const                   = 0;

    // virtual void mymvprod_transp_local_to_local(const T *const in, T *const out, const int &mu = 1, T *work = nullptr) const = 0;
    // virtual void mymvprod_transp_local_to_global(const T *const in, T *const out, const int &mu = 1) const                   = 0;

    // // Special matrix-vector product for building coarse space
    // virtual void mvprod_subrhs(const T *const in, T *const out, const int &mu, const int &offset, const int &size, const int &margin) const = 0;

    // // Local factorisation
    // virtual void copy_local_dense_perm(T *) const = 0;

    //  -- Destructors --
    virtual ~VirtualDistributedOperator(){};

  protected:
    //  -- Constructors --
    VirtualDistributedOperator()                                   = default;
    VirtualDistributedOperator(const VirtualDistributedOperator &) = default;
    VirtualDistributedOperator(VirtualDistributedOperator &&)      = default;

    //  -- Assignment --
    VirtualDistributedOperator &operator=(const VirtualDistributedOperator &) = default;
    VirtualDistributedOperator &operator=(VirtualDistributedOperator &&)      = default;
};

} // namespace htool
#endif
