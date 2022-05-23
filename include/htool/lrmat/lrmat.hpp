#ifndef HTOOL_LRMAT_HPP
#define HTOOL_LRMAT_HPP

#include "../blocks/blocks.hpp"
#include "../clustering/cluster.hpp"
#include "../types/matrix.hpp"
#include "../types/virtual_generator.hpp"
#include "virtual_lrmat_generator.hpp"
#include <cassert>
#include <vector>

namespace htool {

template <typename T>
class Block;

template <typename T>
class VirtualBlockData;

template <typename T>
class LowRankMatrix : public VirtualBlockData<T> {

  protected:
    // Data member
    int rank;
    int nr, nc;
    Matrix<T> U, V;
    double epsilon;

  public:
    // Constructors
    LowRankMatrix(const Block<T> &block, const VirtualGenerator<T> &A, const VirtualLowRankGenerator<T> &LRGenerator, const double *const xt, const double *const xs, int rank0 = -1, double epsilon0 = 1e-3, bool use_permutation = true) : rank(rank0), nr(block.get_target_cluster().get_size()), nc(block.get_source_cluster().get_size()), U(), V(), epsilon(epsilon0) {

        if (this->rank == 0) {
            T *uu, *vv;
            uu = new T[this->nr];
            vv = new T[this->nc];
            std::fill_n(uu, this->nr, 0);
            std::fill_n(vv, this->nc, 0);
            this->U.assign(this->nr, 1, uu, LRGenerator.is_htool_owning_data());
            this->V.assign(1, this->nc, vv, LRGenerator.is_htool_owning_data());
        } else {
            T *uu, *vv;
            if (use_permutation)
                LRGenerator.copy_low_rank_approximation(epsilon, this->nr, this->nc, block.get_target_cluster().get_perm_data(), block.get_source_cluster().get_perm_data(), rank, &uu, &vv, A, block.get_target_cluster(), xt, block.get_source_cluster(), xs);
            else {
                std::vector<int> no_perm_target(block.get_target_cluster().get_size()), no_perm_source(block.get_source_cluster().get_size());
                std::iota(no_perm_target.begin(), no_perm_target.end(), block.get_target_cluster().get_offset());
                std::iota(no_perm_source.begin(), no_perm_source.end(), block.get_source_cluster().get_offset());
                LRGenerator.copy_low_rank_approximation(epsilon, this->nr, this->nc, no_perm_target.data(), no_perm_source.data(), rank, &uu, &vv, A, block.get_target_cluster(), xt, block.get_source_cluster(), xs);
            }

            if (rank > 0) {
                this->U.assign(this->nr, rank, uu, LRGenerator.is_htool_owning_data());
                this->V.assign(rank, this->nc, vv, LRGenerator.is_htool_owning_data());
            } else {
                // rank=-1 will be deleted
            }
        }
    };

    // Getters
    int nb_rows() const { return this->nr; }
    int nb_cols() const { return this->nc; }
    int rank_of() const { return this->rank; }

    T get_U(int i, int j) const { return this->U(i, j); }
    T get_V(int i, int j) const { return this->V(i, j); }
    void assign_U(int i, int j, T *ptr) { return this->U.assign(i, j, ptr); }
    void assign_V(int i, int j, T *ptr) { return this->V.assign(i, j, ptr); }
    std::vector<int> get_xr() const { return this->xr; }
    std::vector<int> get_xc() const { return this->xc; }
    double get_epsilon() const { return this->epsilon; }

    std::vector<T> operator*(const std::vector<T> &a) const {
        return this->U * (this->V * a);
    }
    void mvprod(const T *const in, T *const out) const {
        if (rank == 0) {
            std::fill(out, out + U.nb_cols(), 0);
        } else {
            std::vector<T> a(this->rank);
            V.mvprod(in, a.data());
            U.mvprod(a.data(), out);
        }
    }

    void add_mvprod_row_major(const T *const in, T *const out, const int &mu, char transb = 'T', char op = 'N') const override {
        if (rank != 0) {
            std::vector<T> a(this->rank * mu);
            if (op == 'N') {
                V.mvprod_row_major(in, a.data(), mu, transb, op);
                U.add_mvprod_row_major(a.data(), out, mu, transb, op);
            } else if (op == 'C' || op == 'T') {
                U.mvprod_row_major(in, a.data(), mu, transb, op);
                V.add_mvprod_row_major(a.data(), out, mu, transb, op);
            }
        }
    }

    void get_whole_matrix(T *const out) const {
        char transa = 'N';
        char transb = 'N';
        int M       = U.nb_rows();
        int N       = V.nb_cols();
        int K       = U.nb_cols();
        T alpha     = 1;
        int lda     = U.nb_rows();
        int ldb     = V.nb_rows();
        T beta      = 0;
        int ldc     = U.nb_rows();

        Blas<T>::gemm(&transa, &transb, &M, &N, &K, &alpha, &(U(0, 0)), &lda, &(V(0, 0)), &ldb, &beta, out, &ldc);
    }

    double compression_ratio() const {
        return (nr * nc) / (double)(this->rank * (nr + nc));
    }

    double space_saving() const {
        return (1 - (this->rank * (1. / double(nr) + 1. / double(nc))));
    }

    friend std::ostream &operator<<(std::ostream &os, const LowRankMatrix &m) {
        os << "rank:\t" << m.rank << std::endl;
        os << "nr:\t" << m.nr << std::endl;
        os << "nc:\t" << m.nc << std::endl;
        os << "U:\n";
        os << m.U << std::endl;
        os << m.V << std::endl;

        return os;
    }
};

template <typename T>
underlying_type<T> Frobenius_relative_error(const Block<T> &block, const LowRankMatrix<T> &lrmat, const VirtualGenerator<T> &ref, int reqrank = -1) {
    assert(reqrank <= lrmat.rank_of());
    if (reqrank == -1) {
        reqrank = lrmat.rank_of();
    }
    underlying_type<T> norm = 0;
    underlying_type<T> err  = 0;
    std::vector<T> aux(lrmat.nb_rows() * lrmat.nb_cols());
    ref.copy_submatrix(lrmat.nb_rows(), lrmat.nb_cols(), block.get_target_cluster().get_perm_data(), block.get_source_cluster().get_perm_data(), aux.data());
    for (int j = 0; j < lrmat.nb_rows(); j++) {
        for (int k = 0; k < lrmat.nb_cols(); k++) {
            norm += std::pow(std::abs(aux[j + k * lrmat.nb_rows()]), 2);
            for (int l = 0; l < reqrank; l++) {
                aux[j + k * lrmat.nb_rows()] = aux[j + k * lrmat.nb_rows()] - lrmat.get_U(j, l) * lrmat.get_V(l, k);
            }
            err += std::pow(std::abs(aux[j + k * lrmat.nb_rows()]), 2);
        }
    }
    err = err / norm;
    return std::sqrt(err);
}

template <typename T>
underlying_type<T> Frobenius_absolute_error(const Block<T> &block, const LowRankMatrix<T> &lrmat, const VirtualGenerator<T> &ref, int reqrank = -1) {
    assert(reqrank <= lrmat.rank_of());
    if (reqrank == -1) {
        reqrank = lrmat.rank_of();
    }
    underlying_type<T> err = 0;
    std::vector<T> aux(lrmat.nb_rows() * lrmat.nb_cols());
    ref.copy_submatrix(lrmat.nb_rows(), lrmat.nb_cols(), block.get_target_cluster().get_perm_data(), block.get_source_cluster().get_perm_data(), aux.data());

    for (int j = 0; j < lrmat.nb_rows(); j++) {
        for (int k = 0; k < lrmat.nb_cols(); k++) {

            for (int l = 0; l < reqrank; l++) {
                aux[j + k * lrmat.nb_rows()] = aux[j + k * lrmat.nb_rows()] - lrmat.get_U(j, l) * lrmat.get_V(l, k);
            }
            err += std::pow(std::abs(aux[j + k * lrmat.nb_rows()]), 2);
        }
    }
    return std::sqrt(err);
}

} // namespace htool

#endif
