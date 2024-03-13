
#ifndef HTOOL_TESTING_GENERATOR_TEST_HPP
#define HTOOL_TESTING_GENERATOR_TEST_HPP

#include "../clustering/cluster_node.hpp"
#include "../hmatrix/interfaces/virtual_generator.hpp"
#include "../matrix/matrix.hpp"
#include <vector>

namespace htool {

template <typename CoefficientPrecision, typename CoordinatePrecision = underlying_type<CoefficientPrecision>>
class GeneratorTest : public VirtualGenerator<CoefficientPrecision> {
  protected:
    int m_target_size, m_source_size;
    const std::vector<CoordinatePrecision> &p1;
    const std::vector<CoordinatePrecision> &p2;
    const Cluster<CoordinatePrecision> &m_target_cluster;
    const Cluster<CoordinatePrecision> &m_source_cluster;
    bool m_use_target_permutation{true};
    bool m_use_source_permutation{true};
    int space_dim;

  public:
    explicit GeneratorTest(int space_dim0, int nr0, int nc0, const std::vector<underlying_type<CoefficientPrecision>> &p10, const std::vector<underlying_type<CoefficientPrecision>> &p20, const Cluster<CoordinatePrecision> &target_cluster, const Cluster<CoordinatePrecision> &source_cluster, bool use_target_permutation, bool use_source_permutation) : m_target_size(nr0), m_source_size(nc0), p1(p10), p2(p20), m_target_cluster(target_cluster), m_source_cluster(source_cluster), m_use_target_permutation(use_target_permutation), m_use_source_permutation(use_source_permutation), space_dim(space_dim0) {}

    virtual CoefficientPrecision get_coef(const int &i, const int &j) const = 0;

    void copy_submatrix(int M, int N, int row_offset, int col_offset, CoefficientPrecision *ptr) const override {
        if (m_use_target_permutation && m_use_source_permutation) {
            const auto &target_permutation = m_target_cluster.get_permutation();
            const auto &source_permutation = m_source_cluster.get_permutation();
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < N; j++) {
                    ptr[i + M * j] = this->get_coef(target_permutation[i + row_offset], source_permutation[j + col_offset]);
                }
            }
        } else if (m_use_target_permutation) {
            const auto &target_permutation = m_target_cluster.get_permutation();
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < N; j++) {
                    ptr[i + M * j] = this->get_coef(target_permutation[i + row_offset], j + col_offset);
                }
            }
        } else if (m_use_source_permutation) {
            const auto &source_permutation = m_source_cluster.get_permutation();
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < N; j++) {
                    ptr[i + M * j] = this->get_coef(i + row_offset, source_permutation[j + col_offset]);
                }
            }
        } else {
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < N; j++) {
                    ptr[i + M * j] = this->get_coef(i + row_offset, j + col_offset);
                }
            }
        }
    }

    void set_use_target_permutation(bool use_target_permutation) { m_use_target_permutation = use_target_permutation; }
    void set_use_source_permutation(bool use_source_permutation) { m_use_source_permutation = use_source_permutation; }

    CoefficientPrecision operator()(int i, int j) { return get_coef(i, j); }

    std::vector<CoefficientPrecision> operator*(std::vector<CoefficientPrecision> &a) const {
        std::vector<CoefficientPrecision> result(m_target_size, 0);
        for (int i = 0; i < m_target_size; i++) {
            for (int k = 0; k < m_source_size; k++) {
                result[i] += this->get_coef(i, k) * a[k];
            }
        }
        return result;
    }
    double normFrob() {
        double norm = 0;
        for (int j = 0; j < this->nb_rows(); j++) {
            for (int k = 0; k < this->nb_cols(); k++) {
                norm = norm + std::pow(std::abs(this->get_coef(j, k)), 2);
            }
        }
        return sqrt(norm);
    }

    void mvprod(const CoefficientPrecision *const in, CoefficientPrecision *const out, const int &mu) const {
        int nr = m_target_size;
        int nc = m_source_size;
        for (int i = 0; i < nr * mu; i++) {
            out[i] = 0;
        }
        for (int m = 0; m < mu; m++) {
            for (int i = 0; i < nr; i++) {
                for (int j = 0; j < nc; j++) {
                    out[nr * m + i] += this->get_coef(i, j) * in[j + m * nc];
                }
            }
        }
    }

    void mvprod_transp(const CoefficientPrecision *const in, CoefficientPrecision *const out, const int &mu) const {
        int nc = m_target_size;
        int nr = m_source_size;
        for (int i = 0; i < nr * mu; i++) {
            out[i] = 0;
        }
        for (int m = 0; m < mu; m++) {
            for (int i = 0; i < nr; i++) {
                for (int j = 0; j < nc; j++) {
                    out[nr * m + i] += this->get_coef(j, i) * in[j + m * nc];
                }
            }
        }
    }

    void mvprod_conj(const CoefficientPrecision *const in, CoefficientPrecision *const out, const int &mu) const {
        int nc = m_target_size;
        int nr = m_source_size;
        for (int i = 0; i < nr * mu; i++) {
            out[i] = 0;
        }
        for (int m = 0; m < mu; m++) {
            for (int i = 0; i < nr; i++) {
                for (int j = 0; j < nc; j++) {
                    out[nr * m + i] += std::conj(this->get_coef(j, i) * in[j + m * nc]);
                }
            }
        }
    }
};

class GeneratorTestDouble : public GeneratorTest<double> {
  public:
    using GeneratorTest::GeneratorTest;
    double get_coef(const int &i, const int &j) const override {
        return 1. / (4 * M_PI * std::sqrt(std::inner_product(p1.begin() + this->space_dim * i, this->p1.begin() + this->space_dim * i + this->space_dim, p2.begin() + this->space_dim * j, double(0), std::plus<double>(), [](double u, double v) { return (u - v) * (u - v); })));
    }
};

class GeneratorTestComplex : public GeneratorTest<std::complex<double>> {
  public:
    using GeneratorTest::GeneratorTest;

    std::complex<double> get_coef(const int &i, const int &j) const override {
        return (1. + std::complex<double>(0, 1)) / (4 * M_PI * std::sqrt(std::inner_product(p1.begin() + this->space_dim * i, this->p1.begin() + this->space_dim * i + this->space_dim, p2.begin() + this->space_dim * j, double(0), std::plus<double>(), [](double u, double v) { return (u - v) * (u - v); })));
    }
};

double sign(double x) {
    if (x > 0)
        return 1;
    if (x < 0)
        return -1;
    return 0;
}

class GeneratorTestDoubleSymmetric : public GeneratorTest<double> {
  public:
    using GeneratorTest::GeneratorTest;

    double get_coef(const int &i, const int &j) const override {
        return 1. / (1e-5 + 4 * M_PI * std::sqrt(std::inner_product(p1.begin() + this->space_dim * i, this->p1.begin() + this->space_dim * i + this->space_dim, p2.begin() + this->space_dim * j, double(0), std::plus<double>(), [](double u, double v) { return (u - v) * (u - v); })));
    }
};

class GeneratorTestComplexSymmetric : public GeneratorTest<std::complex<double>> {
  public:
    using GeneratorTest::GeneratorTest;

    std::complex<double> get_coef(const int &i, const int &j) const override {
        return (1. + std::complex<double>(0, 1)) / (1e-5 + 4 * M_PI * std::sqrt(std::inner_product(p1.begin() + this->space_dim * i, this->p1.begin() + this->space_dim * i + this->space_dim, p2.begin() + this->space_dim * j, double(0), std::plus<double>(), [](double u, double v) { return (u - v) * (u - v); })));
    }
};

class GeneratorTestComplexHermitian : public GeneratorTest<std::complex<double>> {
  public:
    using GeneratorTest::GeneratorTest;

    std::complex<double> get_coef(const int &i, const int &j) const override {
        return (1. + sign(p1[this->space_dim * i] - p2[this->space_dim * j]) * std::complex<double>(0, 1)) / (1e-5 + 4 * M_PI * std::sqrt(std::inner_product(p1.begin() + this->space_dim * i, this->p1.begin() + this->space_dim * i + this->space_dim, p2.begin() + this->space_dim * j, double(0), std::plus<double>(), [](double u, double v) { return (u - v) * (u - v); })));
    }
};

template <typename T>
class GeneratorFromMatrix : public VirtualGeneratorWithPermutation<T> {
    const Matrix<T> &A;

  public:
    explicit GeneratorFromMatrix(const Matrix<T> &A0, const std::vector<int> &target_permutation, const std::vector<int> &source_permutation) : VirtualGeneratorWithPermutation<T>(target_permutation.data(), source_permutation.data()), A(A0) {}

    void copy_submatrix_from_user_numbering(int M, int N, const int *const rows, const int *const cols, T *ptr) const override {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                ptr[i + M * j] = A(rows[i], cols[j]);
            }
        }
    }
};

template <typename T>
class LocalGeneratorFromMatrix : public VirtualGeneratorWithPermutation<T> {
    const Matrix<T> &m_A;
    const std::vector<int> &m_target_local_to_global_numbering;
    const std::vector<int> &m_source_local_to_global_numbering;

  public:
    explicit LocalGeneratorFromMatrix(const Matrix<T> &A, const std::vector<int> &target_permutation, const std::vector<int> &source_permutation, const std::vector<int> &target_local_to_global_numbering, const std::vector<int> &source_local_to_global_numbering) : VirtualGeneratorWithPermutation<T>(target_permutation.data(), source_permutation.data()), m_A(A), m_target_local_to_global_numbering(target_local_to_global_numbering), m_source_local_to_global_numbering(source_local_to_global_numbering) {}

    void copy_submatrix_from_user_numbering(int M, int N, const int *const rows, const int *const cols, T *ptr) const override {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                ptr[i + M * j] = m_A(m_target_local_to_global_numbering[rows[i]], m_source_local_to_global_numbering[cols[j]]);
            }
        }
    }
};
} // namespace htool

#endif
