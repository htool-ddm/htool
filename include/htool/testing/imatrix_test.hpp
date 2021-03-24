
#ifndef HTOOL_TESTING_IMATRIX_HPP
#define HTOOL_TESTING_IMATRIX_HPP

#include "../types/matrix.hpp"
#include <vector>

namespace htool {

template <typename T>
class IMatrixTest : public IMatrix<T> {
  protected:
    const std::vector<double> &p1;
    const std::vector<double> &p2;
    int space_dim;
    T coef;

  public:
    explicit IMatrixTest(int space_dim0, int nr, int nc, const std::vector<double> &p10, const std::vector<double> &p20, T coef0 = 0) : IMatrix<T>(nr, nc), p1(p10), p2(p20), space_dim(space_dim0), coef(coef0) {}

    explicit IMatrixTest(int space_dim0, int nr, const std::vector<double> &p10, T coef0 = 0) : IMatrix<T>(nr, nr), p1(p10), p2(p10), space_dim(space_dim0), coef(coef0) {}

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
};

class IMatrixTestDouble : public IMatrixTest<double> {
  public:
    using IMatrixTest::IMatrixTest;

    double get_coef(const int &i, const int &j) const {
        return 1. / (4 * M_PI * std::sqrt(this->coef + std::inner_product(p1.begin() + this->space_dim * i, this->p1.begin() + this->space_dim * i + this->space_dim, p2.begin() + this->space_dim * j, double(0), std::plus<double>(), [](double u, double v) { return (u - v) * (u - v); })));
    }
};

class IMatrixTestComplex : public IMatrixTest<std::complex<double>> {
  public:
    using IMatrixTest::IMatrixTest;

    std::complex<double> get_coef(const int &i, const int &j) const {
        return (1. + std::complex<double>(0, 1)) / (4 * M_PI * std::sqrt(coef + std::inner_product(p1.begin() + space_dim * i, p1.begin() + space_dim * i + space_dim, p2.begin() + space_dim * j, double(0), std::plus<double>(), [](double u, double v) { return (u - v) * (u - v); })));
    }
};

double sign(int x) {
    if (x > 0)
        return 1;
    if (x < 0)
        return -1;
    return 0;
}
class IMatrixTestComplexHermitian : public IMatrixTest<std::complex<double>> {
  public:
    using IMatrixTest::IMatrixTest;

    std::complex<double> get_coef(const int &i, const int &j) const {
        return (1. + sign(i - j) * std::complex<double>(0, 1)) / (4 * M_PI * std::sqrt(coef + std::inner_product(p1.begin() + space_dim * i, p1.begin() + space_dim * i + space_dim, p2.begin() + space_dim * j, double(0), std::plus<double>(), [](double u, double v) { return (u - v) * (u - v); })));
    }
};

} // namespace htool

#endif