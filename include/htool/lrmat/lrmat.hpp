#ifndef HTOOL_LRMAT_HPP
#define HTOOL_LRMAT_HPP

#include "../clustering/cluster.hpp"
#include "../types/matrix.hpp"
#include "../types/virtual_generator.hpp"
#include "virtual_lrmat_generator.hpp"
#include <cassert>
#include <vector>

namespace htool {

template <typename T>
class LowRankMatrix : public IMatrix<T> {

  protected:
    // Data member
    int rank;
    // nr, nc;
    Matrix<T> U, V;
    std::vector<int> ir;
    std::vector<int> ic;
    int offset_i;
    int offset_j;
    double epsilon;
    unsigned int dimension;

  public:
    // Constructors
    LowRankMatrix() = delete;
    LowRankMatrix(int dimension0, const std::vector<int> &ir0, const std::vector<int> &ic0, int rank0 = -1, double epsilon0 = 1e-3) : IMatrix<T>(dimension0 * ir0.size(), dimension0 * ic0.size()), rank(rank0), U(), V(), ir(ir0), ic(ic0), offset_i(0), offset_j(0), epsilon(epsilon0), dimension(dimension0) {}

    LowRankMatrix(int dimension0, const std::vector<int> &ir0, const std::vector<int> &ic0, int offset_i0, int offset_j0, int rank0 = -1, double epsilon0 = 1e-3) : IMatrix<T>(dimension0 * ir0.size(), dimension0 * ic0.size()), rank(rank0), U(), V(), ir(ir0), ic(ic0), offset_i(offset_i0), offset_j(offset_j0), epsilon(epsilon0), dimension(dimension0) {}

    // VIrtual function
    void build(const VirtualGenerator<T> &A, const VirtualLowRankGenerator<T> &LRGenerator, const VirtualCluster &t, const double *const xt, const VirtualCluster &s, const double *const xs) {
        if (this->rank == 0) {
            T *uu, *vv;
            uu = new T[this->nr];
            vv = new T[this->nc];
            std::fill_n(uu, this->nr, 0);
            std::fill_n(vv, this->nc, 0);
            this->U.assign(this->nr, 1, uu);
            this->V.assign(1, this->nc, vv);
        } else {
            T *uu, *vv;
            LRGenerator.copy_low_rank_approximation(epsilon, ir.size(), ic.size(), ir.data(), ic.data(), rank, &uu, &vv, A, t, xt, s, xs);
            if (rank > 0) {
                this->U.assign(this->nr, rank, uu);
                this->V.assign(rank, this->nc, vv);
            } else {
                // rank=-1 will be deleted
            }
        }
    };

    // Getters
    // int nb_rows() const { return this->nr; }
    // int nb_cols() const { return this->nc; }
    int rank_of() const { return this->rank; }
    std::vector<int> get_ir() const { return this->ir; }
    std::vector<int> get_ic() const { return this->ic; }
    int get_offset_i() const { return this->offset_i; }
    int get_offset_j() const { return this->offset_j; }
    T get_U(int i, int j) const { return this->U(i, j); }
    T get_V(int i, int j) const { return this->V(i, j); }
    void assign_U(int i, int j, T *ptr) { return this->U.assign(i, j, ptr); }
    void assign_V(int i, int j, T *ptr) { return this->V.assign(i, j, ptr); }
    std::vector<int> get_xr() const { return this->xr; }
    std::vector<int> get_xc() const { return this->xc; }
    double get_epsilon() const { return this->epsilon; }
    int get_dimension() const { return this->dimension; }

    void set_epsilon(double epsilon0) { this->epsilon = epsilon0; }

    std::vector<T> operator*(const std::vector<T> &a) const {
        return this->U * (this->V * a);
    }
    void mvprod(const T *const in, T *const out) const {
        if (rank == 0) {
            std::fill(out, out + this->nr, 0);
        } else {
            std::vector<T> a(this->rank);
            V.mvprod(in, a.data());
            U.mvprod(a.data(), out);
        }
    }

    void add_mvprod_row_major(const T *const in, T *const out, const int &mu, char transb = 'T', char op = 'N') const {
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
        return (this->nr * this->nc) / (double)(this->rank * (this->nr + this->nc));
    }

    double space_saving() const {
        return (1 - (this->rank * (1. / double(this->nr) + 1. / double(this->nc))));
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
underlying_type<T> Frobenius_relative_error(const LowRankMatrix<T> &lrmat, const VirtualGenerator<T> &ref, int reqrank = -1) {
    assert(reqrank <= lrmat.rank_of());
    if (reqrank == -1) {
        reqrank = lrmat.rank_of();
    }
    underlying_type<T> norm = 0;
    underlying_type<T> err  = 0;
    std::vector<int> ir     = lrmat.get_ir();
    std::vector<int> ic     = lrmat.get_ic();

    for (int j = 0; j < lrmat.nb_rows(); j++) {
        for (int k = 0; k < lrmat.nb_cols(); k++) {
            T aux;
            ref.copy_submatrix(1, 1, &(ir[j]), &(ic[k]), &aux);
            norm += std::pow(std::abs(aux), 2);
            for (int l = 0; l < reqrank; l++) {
                aux = aux - lrmat.get_U(j, l) * lrmat.get_V(l, k);
            }
            err += std::pow(std::abs(aux), 2);
        }
    }
    err = err / norm;
    return std::sqrt(err);
}

template <typename T>
underlying_type<T> Frobenius_absolute_error(const LowRankMatrix<T> &lrmat, const VirtualGenerator<T> &ref, int reqrank = -1) {
    assert(reqrank <= lrmat.rank_of());
    if (reqrank == -1) {
        reqrank = lrmat.rank_of();
    }
    underlying_type<T> err = 0;
    std::vector<int> ir    = lrmat.get_ir();
    std::vector<int> ic    = lrmat.get_ic();

    for (int j = 0; j < lrmat.nb_rows(); j++) {
        for (int k = 0; k < lrmat.nb_cols(); k++) {
            T aux;
            ref.copy_submatrix(1, 1, &(ir[j]), &(ic[k]), &aux);
            for (int l = 0; l < reqrank; l++) {
                aux = aux - lrmat.get_U(j, l) * lrmat.get_V(l, k);
            }
            err += std::pow(std::abs(aux), 2);
        }
    }
    return std::sqrt(err);
}

} // namespace htool

#endif
