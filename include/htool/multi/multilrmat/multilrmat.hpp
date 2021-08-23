#ifndef HTOOL_MULTI_LRMAT_HPP
#define HTOOL_MULTI_LRMAT_HPP

#include "../clustering/cluster.hpp"
#include "../lrmat/lrmat.hpp"
#include "../multilrmat/virtual_multi_lrmat_generator.hpp"
#include "../types/multimatrix.hpp"
#include <vector>
namespace htool {

template <typename T>
class MultiLowRankMatrix {

  protected:
    // Data member
    int rank, nr, nc, nm;
    // Matrix<T>  U,V;
    std::vector<LowRankMatrix<T>> LowRankMatrices;
    std::vector<int> ir;
    std::vector<int> ic;
    int offset_i;
    int offset_j;

    double epsilon;
    unsigned int dimension;

  public:
    // Constructors
    MultiLowRankMatrix() = delete;
    MultiLowRankMatrix(int dimension0, const std::vector<int> &ir0, const std::vector<int> &ic0, int nm0, int rank0 = -1, double epsilon0 = 1e-3) : rank(rank0), nr(ir0.size()), nc(ic0.size()), nm(nm0), ir(ir0), ic(ic0), offset_i(0), offset_j(0), epsilon(epsilon0), dimension(dimension0) {
        for (int l = 0; l < nm; l++) {
            LowRankMatrices.emplace_back(dimension, ir, ic, offset_i, offset_j, rank0, epsilon);
        }
    }
    MultiLowRankMatrix(int dimension0, const std::vector<int> &ir0, const std::vector<int> &ic0, int nm0, int offset_i0, int offset_j0, int rank0 = -1, double epsilon0 = 1e-3) : rank(rank0), nr(ir0.size()), nc(ic0.size()), nm(nm0), ir(ir0), ic(ic0), offset_i(offset_i0), offset_j(offset_j0), epsilon(epsilon0), dimension(dimension0) {
        for (int l = 0; l < nm; l++) {
            LowRankMatrices.emplace_back(dimension, ir, ic, offset_i, offset_j, rank0, epsilon);
        }
    }

    // VIrtual function
    void build(const MultiIMatrix<T> &A, const VirtualMultiLowRankGenerator<T> &MLRGenerator, const VirtualCluster &t, const double *const xt, const VirtualCluster &s, const double *const xs) {
        if (this->rank == 0) {
            for (int l = 0; l < this->nm; l++) {
                T *uu, *vv;
                uu = new T[this->nr];
                vv = new T[this->nc];
                std::fill_n(uu, this->nr, 0);
                std::fill_n(vv, this->nc, 0);
                this->LowRankMatrices[l].assign_U(this->nr, 1, uu);
                this->LowRankMatrices[l].assign_V(1, this->nc, vv);
            }
        } else {
            T **uu, **vv;
            MLRGenerator.copy_multi_low_rank_approximation(epsilon, ir.size(), ic.size(), ir.data(), ic.data(), rank, &uu, &vv, A, t, xt, s, xs);
            // for (int l = 0; l < this->nm; l++) {
            //     this->LowRankMatrices[l].assign_U(this->nr, rank, uu[l]);
            //     this->LowRankMatrices[l].assign_V(rank, this->nc, vv[l]);
            // }
            // delete[] uu;
            // delete[] vv;
        }
    };

    // Getters
    int nb_rows() const { return this->nr; }
    int nb_cols() const { return this->nc; }
    int nb_lrmats() const { return this->nm; }
    int rank_of() const { return this->rank; }
    std::vector<int> get_ir() const { return this->ir; }
    std::vector<int> get_ic() const { return this->ic; }
    int get_offset_i() const { return this->offset_i; }
    int get_offset_j() const { return this->offset_j; }
    double get_epsilon() const { return this->epsilon; }
    int get_ndofperelt() const { return this->ndofperelt; }

    void set_epsilon(double epsilon0) { this->epsilon = epsilon0; }
    void set_ndofperelt(unsigned int ndofperelt0) { this->ndofperelt = ndofperelt0; }

    LowRankMatrix<T> &operator[](int j) { return LowRankMatrices[j]; };
    const LowRankMatrix<T> &operator[](int j) const { return LowRankMatrices[j]; };
};

template <typename T>
std::vector<double> Frobenius_absolute_error(const MultiLowRankMatrix<T> &lrmat, const MultiIMatrix<T> &ref, int reqrank = -1) {
    assert(reqrank <= lrmat[0].rank_of());
    if (reqrank == -1) {
        reqrank = lrmat[0].rank_of();
    }
    std::vector<T> err(lrmat.nb_lrmats(), 0);
    std::vector<int> ir = lrmat.get_ir();
    std::vector<int> ic = lrmat.get_ic();
    std::vector<T> aux(lrmat.nb_lrmats());

    for (int j = 0; j < lrmat.nb_rows(); j++) {
        for (int k = 0; k < lrmat.nb_cols(); k++) {
            aux = ref.get_coefs(ir[j], ic[k]);
            for (int l = 0; l < lrmat.nb_lrmats(); l++) {
                for (int r = 0; r < reqrank; r++) {
                    aux[l] = aux[l] - lrmat[l].get_U(j, r) * lrmat[l].get_V(r, k);
                }
                err[l] += std::pow(std::abs(aux[l]), 2);
            }
        }
    }

    std::transform(err.begin(), err.end(), err.begin(), (double (*)(double))sqrt);
    return err;
}

template <typename T>
std::vector<double> Frobenius_absolute_error(const MultiLowRankMatrix<std::complex<T>> &lrmat, const MultiIMatrix<std::complex<T>> &ref, int reqrank = -1) {
    assert(reqrank <= lrmat[0].rank_of());

    std::vector<T> err(lrmat.nb_lrmats(), 0);
    std::vector<int> ir = lrmat.get_ir();
    std::vector<int> ic = lrmat.get_ic();
    std::vector<std::complex<T>> aux(lrmat.nb_lrmats());
    for (int j = 0; j < lrmat.nb_rows(); j++) {
        for (int k = 0; k < lrmat.nb_cols(); k++) {
            aux = ref.get_coefs(ir[j], ic[k]);
            for (int l = 0; l < lrmat.nb_lrmats(); l++) {
                if (reqrank == -1) {
                    reqrank = lrmat.rank_of();
                }
                for (int r = 0; r < reqrank; r++) {
                    aux[l] = aux[l] - lrmat[l].get_U(j, r) * lrmat[l].get_V(r, k);
                }
                err[l] += std::pow(std::abs(aux[l]), 2);
            }
        }
    }

    std::transform(err.begin(), err.end(), err.begin(), (double (*)(double))sqrt);
    return err;
}

template <typename T>
double Frobenius_absolute_error(const LowRankMatrix<std::complex<T>> &lrmat, const IMatrix<std::complex<T>> &ref, int reqrank = -1) {
    assert(reqrank <= lrmat.rank_of());
    if (reqrank == -1) {
        reqrank = lrmat.rank_of();
    }
    T err               = 0;
    std::vector<int> ir = lrmat.get_ir();
    std::vector<int> ic = lrmat.get_ic();

    for (int j = 0; j < lrmat.nb_rows(); j++) {
        for (int k = 0; k < lrmat.nb_cols(); k++) {
            std::complex<T> aux = ref.get_coef(ir[j], ic[k]);
            for (int l = 0; l < reqrank; l++) {
                aux = aux - lrmat.get_U(j, l) * lrmat.get_V(l, k);
            }
            err += std::pow(std::abs(aux), 2);
        }
    }
    return std::sqrt(err);
}

template <typename T>
double Frobenius_absolute_error(const LowRankMatrix<T> &lrmat, const MultiIMatrix<T> &ref, int l, int reqrank = -1) {
    assert(reqrank <= lrmat.rank_of());
    if (reqrank == -1) {
        reqrank = lrmat.rank_of();
    }
    T err               = 0;
    std::vector<int> ir = lrmat.get_ir();
    std::vector<int> ic = lrmat.get_ic();

    for (int j = 0; j < lrmat.nb_rows(); j++) {
        for (int k = 0; k < lrmat.nb_cols(); k++) {
            T aux = ref.get_coefs(ir[j], ic[k])[l];
            for (int l = 0; l < reqrank; l++) {
                aux = aux - lrmat.get_U(j, l) * lrmat.get_V(l, k);
            }
            err += std::pow(std::abs(aux), 2);
        }
    }
    return std::sqrt(err);
}

template <typename T>
double Frobenius_absolute_error(const LowRankMatrix<std::complex<T>> &lrmat, const MultiIMatrix<std::complex<T>> &ref, int l, int reqrank = -1) {
    assert(reqrank <= lrmat.rank_of());
    if (reqrank == -1) {
        reqrank = lrmat.rank_of();
    }
    T err               = 0;
    std::vector<int> ir = lrmat.get_ir();
    std::vector<int> ic = lrmat.get_ic();

    for (int j = 0; j < lrmat.nb_rows(); j++) {
        for (int k = 0; k < lrmat.nb_cols(); k++) {
            std::complex<T> aux = ref.get_coefs(ir[j], ic[k])[l];
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
