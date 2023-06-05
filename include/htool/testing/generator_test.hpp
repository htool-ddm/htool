
#ifndef HTOOL_TESTING_GENERATOR_TEST_HPP
#define HTOOL_TESTING_GENERATOR_TEST_HPP

#include "../hmatrix/interfaces/virtual_generator.hpp" // for VirtualGenera...
#include "../matrix/matrix.hpp"                        // for Matrix
#include "htool/misc/misc.hpp"                         // for underlying_type
#include <cmath>                                       // for sqrt, M_PI
#include <complex>                                     // for complex, oper...
#include <functional>                                  // for plus
#include <numeric>                                     // for inner_product
#include <vector>                                      // for vector

namespace htool {

template <typename CoefficientPrecision, typename CoordinatePrecision = underlying_type<CoefficientPrecision>>
class GeneratorTestWithPermutation : public VirtualGenerator<CoefficientPrecision> {
  protected:
    int m_space_dimension;
    const std::vector<CoordinatePrecision> &m_target_points;
    const std::vector<CoordinatePrecision> &m_source_points;

  public:
    GeneratorTestWithPermutation(int space_dim, const std::vector<CoordinatePrecision> &target_points, const std::vector<CoordinatePrecision> &source_points) : m_space_dimension(space_dim), m_target_points(target_points), m_source_points(source_points) {}

    virtual CoefficientPrecision get_coef(const int &i, const int &j) const = 0;
    CoefficientPrecision operator()(int i, int j) { return get_coef(i, j); }
    void copy_submatrix(int M, int N, const int *rows, const int *cols, CoefficientPrecision *ptr) const override {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                ptr[i + M * j] = this->get_coef(rows[i], cols[j]);
            }
        }
    }
};
// template <typename CoefficientPrecision, typename CoordinatePrecision = underlying_type<CoefficientPrecision>>
// class GeneratorTest : public VirtualInternalGenerator<CoefficientPrecision> {
//   protected:
//     int m_target_size, m_source_size;
//     const std::vector<CoordinatePrecision> &m_target_points;
//     const std::vector<CoordinatePrecision> &m_source_points;
//     const Cluster<CoordinatePrecision> &m_target_cluster;
//     const Cluster<CoordinatePrecision> &m_source_cluster;
//     bool m_use_target_permutation{true};
//     bool m_use_source_permutation{true};
//     int space_dim;

//   public:
//     explicit GeneratorTest(int space_dim0, int nr0, int nc0, const std::vector<underlying_type<CoefficientPrecision>> &m_target_points0, const std::vector<underlying_type<CoefficientPrecision>> &m_source_points0, const Cluster<CoordinatePrecision> &target_cluster, const Cluster<CoordinatePrecision> &source_cluster, bool use_target_permutation, bool use_source_permutation) : m_target_size(nr0), m_source_size(nc0), m_target_points(m_target_points0), m_source_points(m_source_points0), m_target_cluster(target_cluster), m_source_cluster(source_cluster), m_use_target_permutation(use_target_permutation), m_use_source_permutation(use_source_permutation), space_dim(space_dim0) {}

//     virtual CoefficientPrecision get_coef(const int &i, const int &j) const = 0;

//     void copy_submatrix(int M, int N, int row_offset, int col_offset, CoefficientPrecision *ptr) const override {
//         if (m_use_target_permutation && m_use_source_permutation) {
//             const auto &target_permutation = m_target_cluster.get_permutation();
//             const auto &source_permutation = m_source_cluster.get_permutation();
//             for (int i = 0; i < M; i++) {
//                 for (int j = 0; j < N; j++) {
//                     ptr[i + M * j] = this->get_coef(target_permutation[i + row_offset], source_permutation[j + col_offset]);
//                 }
//             }
//         } else if (m_use_target_permutation) {
//             const auto &target_permutation = m_target_cluster.get_permutation();
//             for (int i = 0; i < M; i++) {
//                 for (int j = 0; j < N; j++) {
//                     ptr[i + M * j] = this->get_coef(target_permutation[i + row_offset], j + col_offset);
//                 }
//             }
//         } else if (m_use_source_permutation) {
//             const auto &source_permutation = m_source_cluster.get_permutation();
//             for (int i = 0; i < M; i++) {
//                 for (int j = 0; j < N; j++) {
//                     ptr[i + M * j] = this->get_coef(i + row_offset, source_permutation[j + col_offset]);
//                 }
//             }
//         } else {
//             for (int i = 0; i < M; i++) {
//                 for (int j = 0; j < N; j++) {
//                     ptr[i + M * j] = this->get_coef(i + row_offset, j + col_offset);
//                 }
//             }
//         }
//     }

//     void set_use_target_permutation(bool use_target_permutation) { m_use_target_permutation = use_target_permutation; }
//     void set_use_source_permutation(bool use_source_permutation) { m_use_source_permutation = use_source_permutation; }

//     CoefficientPrecision operator()(int i, int j) { return get_coef(i, j); }

//     std::vector<CoefficientPrecision> operator*(std::vector<CoefficientPrecision> &a) const {
//         std::vector<CoefficientPrecision> result(m_target_size, 0);
//         for (int i = 0; i < m_target_size; i++) {
//             for (int k = 0; k < m_source_size; k++) {
//                 result[i] += this->get_coef(i, k) * a[k];
//             }
//         }
//         return result;
//     }
//     double normFrob() {
//         double norm = 0;
//         for (int j = 0; j < this->nb_rows(); j++) {
//             for (int k = 0; k < this->nb_cols(); k++) {
//                 norm = norm + std::pow(std::abs(this->get_coef(j, k)), 2);
//             }
//         }
//         return sqrt(norm);
//     }

//     void mvprod(const CoefficientPrecision *const in, CoefficientPrecision *const out, const int &mu) const {
//         int nr = m_target_size;
//         int nc = m_source_size;
//         for (int i = 0; i < nr * mu; i++) {
//             out[i] = 0;
//         }
//         for (int m = 0; m < mu; m++) {
//             for (int i = 0; i < nr; i++) {
//                 for (int j = 0; j < nc; j++) {
//                     out[nr * m + i] += this->get_coef(i, j) * in[j + m * nc];
//                 }
//             }
//         }
//     }

//     void mvprod_transp(const CoefficientPrecision *const in, CoefficientPrecision *const out, const int &mu) const {
//         int nc = m_target_size;
//         int nr = m_source_size;
//         for (int i = 0; i < nr * mu; i++) {
//             out[i] = 0;
//         }
//         for (int m = 0; m < mu; m++) {
//             for (int i = 0; i < nr; i++) {
//                 for (int j = 0; j < nc; j++) {
//                     out[nr * m + i] += this->get_coef(j, i) * in[j + m * nc];
//                 }
//             }
//         }
//     }

//     void mvprod_conj(const CoefficientPrecision *const in, CoefficientPrecision *const out, const int &mu) const {
//         int nc = m_target_size;
//         int nr = m_source_size;
//         for (int i = 0; i < nr * mu; i++) {
//             out[i] = 0;
//         }
//         for (int m = 0; m < mu; m++) {
//             for (int i = 0; i < nr; i++) {
//                 for (int j = 0; j < nc; j++) {
//                     out[nr * m + i] += std::conj(this->get_coef(j, i) * in[j + m * nc]);
//                 }
//             }
//         }
//     }
// };

class GeneratorTestDouble : public GeneratorTestWithPermutation<double> {
  public:
    using GeneratorTestWithPermutation::GeneratorTestWithPermutation;
    double get_coef(const int &i, const int &j) const override {
        return 1. / (4 * M_PI * std::sqrt(std::inner_product(m_target_points.begin() + m_space_dimension * i, this->m_target_points.begin() + m_space_dimension * i + m_space_dimension, m_source_points.begin() + m_space_dimension * j, double(0), std::plus<double>(), [](double u, double v) { return (u - v) * (u - v); })));
    }
};

class GeneratorTestComplex : public GeneratorTestWithPermutation<std::complex<double>> {
  public:
    using GeneratorTestWithPermutation::GeneratorTestWithPermutation;

    std::complex<double> get_coef(const int &i, const int &j) const override {
        return (1. + std::complex<double>(0, 1)) / (4 * M_PI * std::sqrt(std::inner_product(m_target_points.begin() + m_space_dimension * i, this->m_target_points.begin() + m_space_dimension * i + m_space_dimension, m_source_points.begin() + m_space_dimension * j, double(0), std::plus<double>(), [](double u, double v) { return (u - v) * (u - v); })));
    }
};

double sign(double x) {
    if (x > 0)
        return 1;
    if (x < 0)
        return -1;
    return 0;
}

class GeneratorTestDoubleSymmetric : public GeneratorTestWithPermutation<double> {
  public:
    using GeneratorTestWithPermutation::GeneratorTestWithPermutation;

    double get_coef(const int &i, const int &j) const override {
        return 1. / (1e-5 + 4 * M_PI * std::sqrt(std::inner_product(m_target_points.begin() + m_space_dimension * i, this->m_target_points.begin() + m_space_dimension * i + m_space_dimension, m_source_points.begin() + m_space_dimension * j, double(0), std::plus<double>(), [](double u, double v) { return (u - v) * (u - v); })));
    }
};

class GeneratorTestComplexSymmetric : public GeneratorTestWithPermutation<std::complex<double>> {
  public:
    using GeneratorTestWithPermutation::GeneratorTestWithPermutation;

    std::complex<double> get_coef(const int &i, const int &j) const override {
        return (1. + std::complex<double>(0, 1)) / (1e-5 + 4 * M_PI * std::sqrt(std::inner_product(m_target_points.begin() + m_space_dimension * i, this->m_target_points.begin() + m_space_dimension * i + m_space_dimension, m_source_points.begin() + m_space_dimension * j, double(0), std::plus<double>(), [](double u, double v) { return (u - v) * (u - v); })));
    }
};

class GeneratorTestComplexHermitian : public GeneratorTestWithPermutation<std::complex<double>> {
  public:
    using GeneratorTestWithPermutation::GeneratorTestWithPermutation;

    std::complex<double> get_coef(const int &i, const int &j) const override {
        return (1. + sign(m_target_points[m_space_dimension * i] - m_source_points[m_space_dimension * j]) * std::complex<double>(0, 1)) / (1e-5 + 4 * M_PI * std::sqrt(std::inner_product(m_target_points.begin() + m_space_dimension * i, this->m_target_points.begin() + m_space_dimension * i + m_space_dimension, m_source_points.begin() + m_space_dimension * j, double(0), std::plus<double>(), [](double u, double v) { return (u - v) * (u - v); })));
    }
};

template <typename T>
class GeneratorInUserNumberingFromMatrix : public VirtualGenerator<T> {
  public:
    const Matrix<T> &A;

    GeneratorInUserNumberingFromMatrix(const Matrix<T> &A0) : A(A0) {}

    virtual void copy_submatrix(int M, int N, const int *rows, const int *cols, T *ptr) const override {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                ptr[i + M * j] = A(rows[i], cols[j]);
            }
        }
    }
};

// class GeneratorFromMatrix : public InternalGeneratorWithPermutation<T> {

//   public:
//     explicit GeneratorFromMatrix(const Matrix<T> &A0, const std::vector<int> &target_permutation, const std::vector<int> &source_permutation) : InternalGeneratorWithPermutation<T>(GeneratorInUserNumberingFromMatrix(A0), target_permutation.data(), source_permutation.data()) {}

//   protected:
//     class GeneratorInUserNumberingFromMatrix : VirtualGenerator<T> {
//       public:
//         const Matrix<T> &A;

//         GeneratorInUserNumberingFromMatrix(const Matrix<T> &A0) : A(A0) {}

//         virtual void copy_submatrix(int M, int N, const int *rows, const int *cols, T *ptr) {
//             for (int i = 0; i < M; i++) {
//                 for (int j = 0; j < N; j++) {
//                     ptr[i + M * j] = A(rows[i], cols[j]);
//                 }
//             }
//         }
//     };
// };

template <typename T>
class LocalGeneratorInUserNumberingFromMatrix : public VirtualGenerator<T> {
    const Matrix<T> &m_A;
    const std::vector<int> &m_target_local_to_global_numbering;
    const std::vector<int> &m_source_local_to_global_numbering;

  public:
    LocalGeneratorInUserNumberingFromMatrix(const Matrix<T> &A, const std::vector<int> &target_local_to_global_numbering, const std::vector<int> &source_local_to_global_numbering) : m_A(A), m_target_local_to_global_numbering(target_local_to_global_numbering), m_source_local_to_global_numbering(source_local_to_global_numbering) {}

    void copy_submatrix(int M, int N, const int *const rows, const int *const cols, T *ptr) const override {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                ptr[i + M * j] = m_A(m_target_local_to_global_numbering[rows[i]], m_source_local_to_global_numbering[cols[j]]);
            }
        }
    }
};
// class LocalGeneratorFromMatrix : public InternalGeneratorWithPermutation<T> {
//     const std::vector<int> &m_target_local_to_global_numbering;
//     const std::vector<int> &m_source_local_to_global_numbering;

//   public:
//     explicit LocalGeneratorFromMatrix(const Matrix<T> &A, const std::vector<int> &target_permutation, const std::vector<int> &source_permutation, const std::vector<int> &target_local_to_global_numbering, const std::vector<int> &source_local_to_global_numbering) : InternalGeneratorWithPermutation<T>(LocalGeneratorInUserNumberingFromMatrix(A), target_permutation.data(), source_permutation.data()), m_target_local_to_global_numbering(target_local_to_global_numbering), m_source_local_to_global_numbering(source_local_to_global_numbering) {}

//   private:
//     class LocalGeneratorInUserNumberingFromMatrix : VirtualGenerator<T> {
//       public:
//         const Matrix<T> &m_A;
//         LocalGeneratorInUserNumberingFromMatrix(const Matrix<T> &A) : m_A(A) {}
//         void copy_submatrix_from(int M, int N, const int *const rows, const int *const cols, T *ptr) const override {
//             for (int i = 0; i < M; i++) {
//                 for (int j = 0; j < N; j++) {
//                     ptr[i + M * j] = m_A(m_target_local_to_global_numbering[rows[i]], m_source_local_to_global_numbering[cols[j]]);
//                 }
//             }
//         }
//     };
// };
} // namespace htool

#endif
