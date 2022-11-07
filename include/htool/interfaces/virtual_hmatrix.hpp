#ifndef HTOOL_HMATRIX_VIRTUAL_HPP
#define HTOOL_HMATRIX_VIRTUAL_HPP

#include "virtual_generator.hpp"
#include "virtual_off_diagonal_approximation.hpp"
#include <map>
#include <mpi.h>
#include <vector>

namespace htool {
template <class T>
class VirtualHMatrix {
  public:
    // Getters
    virtual int nb_rows() const                                    = 0;
    virtual int nb_cols() const                                    = 0;
    virtual MPI_Comm get_comm() const                              = 0;
    virtual int get_rankworld() const                              = 0;
    virtual int get_sizeworld() const                              = 0;
    virtual int get_local_size() const                             = 0;
    virtual int get_local_offset() const                           = 0;
    virtual char get_symmetry_type() const                         = 0;
    virtual char get_storage_type() const                          = 0;
    virtual std::vector<T> get_local_diagonal(bool = true) const   = 0;
    virtual void copy_local_diagonal(T *, bool = true) const       = 0;
    virtual Matrix<T> get_local_diagonal_block(bool = true) const  = 0;
    virtual void copy_local_diagonal_block(T *, bool = true) const = 0;

    virtual const VirtualCluster *get_target_cluster() const = 0;
    virtual const VirtualCluster *get_source_cluster() const = 0;

    // Getters/setters for parameters
    virtual double get_epsilon() const     = 0;
    virtual double get_eta() const         = 0;
    virtual int get_dimension() const      = 0;
    virtual int get_minsourcedepth() const = 0;
    virtual int get_mintargetdepth() const = 0;
    virtual int get_maxblocksize() const   = 0;

    virtual void set_epsilon(double epsilon0)                                            = 0;
    virtual void set_eta(double eta0)                                                    = 0;
    virtual void set_maxblocksize(unsigned int maxblocksize)                             = 0;
    virtual void set_minsourcedepth(unsigned int minsourcedepth0)                        = 0;
    virtual void set_mintargetdepth(unsigned int mintargetdepth0)                        = 0;
    virtual void set_use_permutation(bool choice)                                        = 0;
    virtual void set_compression(std::shared_ptr<VirtualLowRankGenerator<T>> compressor) = 0;

    // functions for user-defined off diagonal
    virtual void get_off_diagonal_size(int &nr_off_diagonal, int &nc_off_diagonal) const                                                                           = 0;
    virtual void get_off_diagonal_geometries(const double *target_points, const double *source_points, double *new_target_points, double *new_source_points) const = 0;
    virtual void set_off_diagonal_approximation(std::shared_ptr<VirtualOffDiagonalApproximation<T>> OffDiagonalApproximation)                                      = 0;

    // Build
    virtual void build(VirtualGenerator<T> &mat, const double *const xt, const double *const xs) = 0;
    virtual void build(VirtualGenerator<T> &mat, const double *const xt)                         = 0;

    // Mat vec prod
    virtual void mvprod_global_to_global(const T *const in, T *const out, const int &mu = 1) const                  = 0;
    virtual void mvprod_local_to_local(const T *const in, T *const out, const int &mu = 1, T *work = nullptr) const = 0;

    virtual void mvprod_transp_global_to_global(const T *const in, T *const out, const int &mu = 1) const                  = 0;
    virtual void mvprod_transp_local_to_local(const T *const in, T *const out, const int &mu = 1, T *work = nullptr) const = 0;

    virtual void mymvprod_local_to_local(const T *const in, T *const out, const int &mu = 1, T *work = nullptr) const = 0;
    virtual void mymvprod_global_to_local(const T *const in, T *const out, const int &mu = 1) const                   = 0;

    virtual void mymvprod_transp_local_to_local(const T *const in, T *const out, const int &mu = 1, T *work = nullptr) const = 0;
    virtual void mymvprod_transp_local_to_global(const T *const in, T *const out, const int &mu = 1) const                   = 0;

    virtual void mvprod_subrhs(const T *const in, T *const out, const int &mu, const int &offset, const int &size, const int &margin) const = 0;

    // Infos
    virtual const std::map<std::string, std::string> &get_infos() const = 0;
    virtual std::string get_infos(const std::string &key) const         = 0;
    virtual void print_infos() const                                    = 0;

    // Convert
    virtual Matrix<T> get_local_dense_perm() const = 0;
    virtual void copy_local_dense_perm(T *) const  = 0;

    virtual ~VirtualHMatrix(){};
};

} // namespace htool
#endif
