
#ifndef HTOOL_TESTING_GENERATOR_TEST_HPP
#define HTOOL_TESTING_GENERATOR_TEST_HPP

#include "../types/matrix.hpp"
#include <vector>

namespace htool {

template <typename T>
class GeneratorTest : public VirtualGenerator<T> {
  protected:
    const std::vector<double> &p1;
    const std::vector<double> &p2;
    int space_dim;

  public:
    explicit GeneratorTest(int space_dim0, int nr, int nc, const std::vector<double> &p10, const std::vector<double> &p20) : VirtualGenerator<T>(nr, nc), p1(p10), p2(p20), space_dim(space_dim0) {}

    explicit GeneratorTest(int space_dim0, int nr, const std::vector<double> &p10) : VirtualGenerator<T>(nr, nr), p1(p10), p2(p10), space_dim(space_dim0) {}

    virtual T get_coef(const int &i, const int &j) const = 0;

    std::vector<T>
    operator*(std::vector<T> &a) const {
        std::vector<T> result(this->nr, 0);
        for (int i = 0; i < this->nr; i++) {
            for (int k = 0; k < this->nc; k++) {
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

    void mvprod(const T *const in, T *const out, const int &mu) const {
        int nr = this->nr;
        int nc = this->nc;
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

    void mvprod_transp(const T *const in, T *const out, const int &mu) const {
        int nc = this->nr;
        int nr = this->nc;
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

    void mvprod_conj(const T *const in, T *const out, const int &mu) const {
        int nc = this->nr;
        int nr = this->nc;
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

    void copy_submatrix(int M, int N, const int *const rows, const int *const cols, double *ptr) const override {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                ptr[i + M * j] = this->get_coef(rows[i], cols[j]);
            }
        }
    }
};

class GeneratorTestComplex : public GeneratorTest<std::complex<double>> {
  public:
    using GeneratorTest::GeneratorTest;

    std::complex<double> get_coef(const int &i, const int &j) const override {
        return (1. + std::complex<double>(0, 1)) / (4 * M_PI * std::sqrt(std::inner_product(p1.begin() + this->space_dim * i, this->p1.begin() + this->space_dim * i + this->space_dim, p2.begin() + this->space_dim * j, double(0), std::plus<double>(), [](double u, double v) { return (u - v) * (u - v); })));
    }

    void copy_submatrix(int M, int N, const int *const rows, const int *const cols, std::complex<double> *ptr) const override {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                ptr[i + M * j] = this->get_coef(rows[i], cols[j]);
            }
        }
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

    void copy_submatrix(int M, int N, const int *const rows, const int *const cols, double *ptr) const override {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                ptr[i + M * j] = this->get_coef(rows[i], cols[j]);
            }
        }
    }
};

class GeneratorTestComplexSymmetric : public GeneratorTest<std::complex<double>> {
  public:
    using GeneratorTest::GeneratorTest;

    std::complex<double> get_coef(const int &i, const int &j) const override {
        return (1. + std::complex<double>(0, 1)) / (1e-5 + 4 * M_PI * std::sqrt(std::inner_product(p1.begin() + this->space_dim * i, this->p1.begin() + this->space_dim * i + this->space_dim, p2.begin() + this->space_dim * j, double(0), std::plus<double>(), [](double u, double v) { return (u - v) * (u - v); })));
    }

    void copy_submatrix(int M, int N, const int *const rows, const int *const cols, std::complex<double> *ptr) const override {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                ptr[i + M * j] = this->get_coef(rows[i], cols[j]);
            }
        }
    }
};

class GeneratorTestComplexHermitian : public GeneratorTest<std::complex<double>> {
  public:
    using GeneratorTest::GeneratorTest;

    std::complex<double> get_coef(const int &i, const int &j) const override {
        return (1. + sign(p1[this->space_dim * i] - p2[this->space_dim * j]) * std::complex<double>(0, 1)) / (1e-5 + 4 * M_PI * std::sqrt(std::inner_product(p1.begin() + this->space_dim * i, this->p1.begin() + this->space_dim * i + this->space_dim, p2.begin() + this->space_dim * j, double(0), std::plus<double>(), [](double u, double v) { return (u - v) * (u - v); })));
    }

    void copy_submatrix(int M, int N, const int *const rows, const int *const cols, std::complex<double> *ptr) const override {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                ptr[i + M * j] = this->get_coef(rows[i], cols[j]);
            }
        }
    }
};

template <typename T>
class GeneratorFromMatrix : public VirtualGenerator<T> {
    const Matrix<T> &A;

  public:
    explicit GeneratorFromMatrix(const Matrix<T> &A0) : VirtualGenerator<T>(A0.nb_rows(), A0.nb_cols()), A(A0) {}

    void copy_submatrix(int M, int N, const int *const rows, const int *const cols, T *ptr) const override {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                ptr[i + M * j] = A(rows[i], cols[j]);
            }
        }
    }
};

} // namespace htool

#endif
