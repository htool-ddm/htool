#ifndef HTOOL_LRMAT_HPP
#define HTOOL_LRMAT_HPP

#include "../../clustering/cluster_node.hpp"
#include "../../matrix/matrix.hpp"
#include "../interfaces/virtual_generator.hpp"
#include "../interfaces/virtual_lrmat_generator.hpp"
#include <cassert>
#include <vector>

namespace htool {

template <typename CoefficientPrecision, typename CoordinatesPrecision = underlying_type<CoefficientPrecision>>
class LowRankMatrix {

  protected:
    Matrix<CoefficientPrecision> m_U, m_V;
    underlying_type<CoefficientPrecision> m_epsilon;

  public:
    // Constructors
    LowRankMatrix(const VirtualGenerator<CoefficientPrecision> &A, const VirtualLowRankGenerator<CoefficientPrecision, CoordinatesPrecision> &LRGenerator, const Cluster<CoordinatesPrecision> &target_cluster, const Cluster<CoordinatesPrecision> &source_cluster, int rank = -1, underlying_type<CoefficientPrecision> epsilon = 1e-3) : m_U(), m_V(), m_epsilon(epsilon) {

        if (rank == 0) {
            m_U.resize(target_cluster.get_size(), 0);
            m_V.resize(0, source_cluster.get_size());
        } else {
            LRGenerator.copy_low_rank_approximation(A, target_cluster, source_cluster, epsilon, rank, m_U, m_V);
        }
    }

    LowRankMatrix(underlying_type<CoefficientPrecision> epsilon) : m_U(), m_V(), m_epsilon(epsilon) {}

    LowRankMatrix &operator=(const LowRankMatrix &rhs) = default;
    LowRankMatrix(const LowRankMatrix &rhs)            = default;

    // Getters
    int nb_rows() const { return m_U.nb_rows(); }
    int nb_cols() const { return m_V.nb_cols(); }
    int rank_of() const { return m_U.nb_cols(); }

    Matrix<CoefficientPrecision> &get_U() { return m_U; }
    Matrix<CoefficientPrecision> &get_V() { return m_V; }
    const Matrix<CoefficientPrecision> &get_U() const { return m_U; }
    const Matrix<CoefficientPrecision> &get_V() const { return m_V; }
    CoefficientPrecision get_U(int i, int j) const { return m_U(i, j); }
    CoefficientPrecision get_V(int i, int j) const { return m_V(i, j); }
    void assign_U(int i, int j, CoefficientPrecision *ptr) { return m_U.assign(i, j, ptr); }
    void assign_V(int i, int j, CoefficientPrecision *ptr) { return m_V.assign(i, j, ptr); }
    underlying_type<CoefficientPrecision> get_epsilon() const { return m_epsilon; }

    std::vector<CoefficientPrecision> operator*(const std::vector<CoefficientPrecision> &a) const {
        return m_U * (m_V * a);
    }

    void add_vector_product(char trans, CoefficientPrecision alpha, const CoefficientPrecision *in, CoefficientPrecision beta, CoefficientPrecision *out) const {
        if (m_U.nb_cols() == 0) {
            std::fill(out, out + m_U.nb_cols(), 0);
        } else if (trans == 'N') {
            std::vector<CoefficientPrecision> a(m_U.nb_cols());
            m_V.add_vector_product(trans, 1, in, 0, a.data());
            m_U.add_vector_product(trans, alpha, a.data(), beta, out);
        } else {
            std::vector<CoefficientPrecision> a(m_U.nb_cols());
            m_U.add_vector_product(trans, 1, in, 0, a.data());
            m_V.add_vector_product(trans, alpha, a.data(), beta, out);
        }
    }

    void add_matrix_product(char transa, CoefficientPrecision alpha, const CoefficientPrecision *in, CoefficientPrecision beta, CoefficientPrecision *out, int mu) const {
        if (m_U.nb_cols() == 0) {
            std::fill(out, out + m_V.nb_cols() * mu, 0);
        } else if (transa == 'N') {
            std::vector<CoefficientPrecision> a(m_U.nb_cols() * mu);
            m_V.add_matrix_product(transa, 1, in, 0, a.data(), mu);
            m_U.add_matrix_product(transa, alpha, a.data(), beta, out, mu);
        } else {
            std::vector<CoefficientPrecision> a(m_U.nb_cols() * mu);
            m_U.add_matrix_product(transa, 1, in, 0, a.data(), mu);
            m_V.add_matrix_product(transa, alpha, a.data(), beta, out, mu);
        }
    }

    void add_matrix_product_row_major(char transa, CoefficientPrecision alpha, const CoefficientPrecision *in, CoefficientPrecision beta, CoefficientPrecision *out, int mu) const {
        if (m_U.nb_cols() == 0) {
            std::fill(out, out + m_V.nb_cols() * mu, 0);
        } else if (transa == 'N') {
            std::vector<CoefficientPrecision> a(m_U.nb_cols() * mu);
            m_V.add_matrix_product_row_major(transa, 1, in, 0, a.data(), mu);
            m_U.add_matrix_product_row_major(transa, alpha, a.data(), beta, out, mu);
        } else {
            std::vector<CoefficientPrecision> a(m_U.nb_cols() * mu);
            m_U.add_matrix_product_row_major(transa, 1, in, 0, a.data(), mu);
            m_V.add_matrix_product_row_major(transa, alpha, a.data(), beta, out, mu);
        }
    }

    void
    mvprod(const CoefficientPrecision *const in, CoefficientPrecision *const out) const {
        if (m_U.nb_cols() == 0) {
            std::fill(out, out + m_U.nb_cols(), 0);
        } else {
            std::vector<CoefficientPrecision> a(m_U.nb_cols());
            m_V.mvprod(in, a.data());
            m_U.mvprod(a.data(), out);
        }
    }

    void add_mvprod_row_major(const CoefficientPrecision *const in, CoefficientPrecision *const out, const int &mu, char transb = 'T', char op = 'N') const {
        if (m_U.nb_cols() != 0) {
            std::vector<CoefficientPrecision> a(m_U.nb_cols() * mu);
            if (op == 'N') {
                m_V.mvprod_row_major(in, a.data(), mu, transb, op);
                m_U.add_mvprod_row_major(a.data(), out, mu, transb, op);
            } else if (op == 'C' || op == 'T') {
                m_U.mvprod_row_major(in, a.data(), mu, transb, op);
                m_V.add_mvprod_row_major(a.data(), out, mu, transb, op);
            }
        }
    }

    void copy_to_dense(CoefficientPrecision *const out) const {
        char transa                = 'N';
        char transb                = 'N';
        int M                      = m_U.nb_rows();
        int N                      = m_V.nb_cols();
        int K                      = m_U.nb_cols();
        CoefficientPrecision alpha = 1;
        int lda                    = m_U.nb_rows();
        int ldb                    = m_V.nb_rows();
        CoefficientPrecision beta  = 0;
        int ldc                    = m_U.nb_rows();

        Blas<CoefficientPrecision>::gemm(&transa, &transb, &M, &N, &K, &alpha, m_U.data(), &lda, m_V.data(), &ldb, &beta, out, &ldc);
    }

    double compression_ratio() const {
        return (m_U.nb_rows() * m_V.nb_cols()) / (double)(m_U.nb_cols() * (m_U.nb_rows() + m_V.nb_cols()));
    }

    double space_saving() const {
        return (1 - (m_U.nb_cols() * (1. / double(m_U.nb_rows()) + 1. / double(m_V.nb_cols()))));
    }

    friend std::ostream &operator<<(std::ostream &os, const LowRankMatrix &m) {
        os << "rank:\t" << m.rank << std::endl;
        os << "number_of_rows:\t" << m.nb_rows() << std::endl;
        os << "m_number_of_columns:\t" << m.nb_cols() << std::endl;
        os << "U:\n";
        os << m.m_U << std::endl;
        os << m.m_V << std::endl;

        return os;
    }
};

template <typename CoefficientPrecision, typename CoordinatePrecision>
underlying_type<CoefficientPrecision> Frobenius_relative_error(const Cluster<CoordinatePrecision> &target_cluster, const Cluster<CoordinatePrecision> &source_cluster, const LowRankMatrix<CoefficientPrecision, CoordinatePrecision> &lrmat, const VirtualGenerator<CoefficientPrecision> &ref, int reqrank = -1) {
    assert(reqrank <= lrmat.rank_of());
    if (reqrank == -1) {
        reqrank = lrmat.rank_of();
    }
    underlying_type<CoefficientPrecision> norm = 0;
    underlying_type<CoefficientPrecision> err  = 0;
    std::vector<CoefficientPrecision> aux(lrmat.nb_rows() * lrmat.nb_cols());
    ref.copy_submatrix(target_cluster, source_cluster, aux.data());
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

template <typename CoefficientPrecision, typename CoordinatePrecision>
underlying_type<CoefficientPrecision> Frobenius_absolute_error(const Cluster<CoordinatePrecision> &target_cluster, const Cluster<CoordinatePrecision> &source_cluster, const LowRankMatrix<CoefficientPrecision, CoordinatePrecision> &lrmat, const VirtualGenerator<CoefficientPrecision> &ref, int reqrank = -1) {
    assert(reqrank <= lrmat.rank_of());
    if (reqrank == -1) {
        reqrank = lrmat.rank_of();
    }
    underlying_type<CoefficientPrecision> err = 0;
    Matrix<CoefficientPrecision> aux(lrmat.nb_rows(), lrmat.nb_cols());
    ref.copy_submatrix(target_cluster.get_size(), source_cluster.get_size(), target_cluster.get_offset(), source_cluster.get_offset(), aux.data());

    for (int j = 0; j < lrmat.nb_rows(); j++) {
        for (int k = 0; k < lrmat.nb_cols(); k++) {

            for (int l = 0; l < reqrank; l++) {
                aux(j, k) = aux(j, k) - lrmat.get_U(j, l) * lrmat.get_V(l, k);
            }
            err += std::pow(std::abs(aux(j, k)), 2);
        }
    }
    return std::sqrt(err);
}

} // namespace htool

#endif
