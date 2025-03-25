#ifndef HTOOL_EVP_HPP
#define HTOOL_EVP_HPP

#include "../basic_types/vector.hpp" // for dprod
#include "../matrix/matrix.hpp"      // for Matrix
#include <array>                     // for array
#include <limits>                    // for numeric_limits
#include <vector>                    // for vector

namespace htool {

template <typename T>
Matrix<T> solve_EVP_2(const Matrix<T> &cov) {
    std::vector<T> dir(2, 0);
    std::vector<T> eigs(2);
    Matrix<T> I(2, 2);
    Matrix<T> result(2, 2);
    I(0, 0) = 1;
    I(1, 1) = 1;
    Matrix<T> prod(2, 2);
    T trace = cov(0, 0) + cov(1, 1);
    T det   = cov(0, 0) * cov(1, 1) - cov(0, 1) * cov(1, 0);
    eigs[0] = trace / static_cast<T>(2.) + std::sqrt((trace * trace / static_cast<T>(4.) - det));
    eigs[1] = trace / static_cast<T>(2.) - std::sqrt((trace * trace / static_cast<T>(4.) - det));
    if (std::abs(eigs[0]) > std::numeric_limits<T>::epsilon()) {
        for (int index : {0, 1}) {
            prod      = (cov - eigs[(index + 1) % 2] * I);
            int ind   = 0;
            T dirnorm = 0;
            do {
                result(0, index) = prod(0, ind);
                result(1, index) = prod(1, ind);
                dirnorm          = std::sqrt(result(0, index) * result(0, index) + result(1, index) * result(1, index));
                ind++;
            } while ((dirnorm < std::numeric_limits<T>::epsilon()) && (ind < 2));
            if (dirnorm < std::numeric_limits<T>::epsilon()) {
                result(0, index) = 1;
                result(1, index) = 0;
            } else {
                result(0, index) /= dirnorm;
                result(1, index) /= dirnorm;
            }
        }
    } else {
        result(0, 0) = 1;
        result(1, 1) = 1;
    }
    return result;
}

template <typename T>
Matrix<T> solve_EVP_3(const Matrix<T> &cov) {
    Matrix<T> result(3, 3);
    T p1 = std::pow(cov(0, 1), 2) + std::pow(cov(0, 2), 2) + std::pow(cov(1, 2), 2);
    std::vector<T> eigs(3);
    Matrix<T> I(3, 3);
    I(0, 0) = 1;
    I(1, 1) = 1;
    I(2, 2) = 1;
    Matrix<T> prod(3, 3);
    if (p1 < std::numeric_limits<T>::epsilon()) {
        // cov is diagonal.
        eigs[0]                    = cov(0, 0);
        eigs[1]                    = cov(1, 1);
        eigs[2]                    = cov(2, 2);
        std::array<int, 3> indexes = {0, 1, 2};
        std::sort(indexes.begin(), indexes.end(), [&eigs](int a, int b) { return eigs[a] < eigs[b]; });
        result(indexes[2], 0) = 1;
        result(indexes[1], 1) = 1;
        result(indexes[0], 2) = 1;
    } else {
        T q  = (cov(0, 0) + cov(1, 1) + cov(2, 2)) / static_cast<T>(3.);
        T p2 = static_cast<T>(std::pow(cov(0, 0) - q, 2)) + static_cast<T>(std::pow(cov(1, 1) - q, 2)) + static_cast<T>(std::pow(cov(2, 2) - q, 2)) + static_cast<T>(2.) * p1;
        T p  = std::sqrt(p2 / static_cast<T>(6.));
        Matrix<T> B(3, 3);
        B      = (static_cast<T>(1) / p) * (cov - q * I);
        T detB = B(0, 0) * (B(1, 1) * B(2, 2) - B(1, 2) * B(2, 1))
                 - B(0, 1) * (B(1, 0) * B(2, 2) - B(1, 2) * B(2, 0))
                 + B(0, 2) * (B(1, 0) * B(2, 1) - B(1, 1) * B(2, 0));
        T r = detB / static_cast<T>(2.);

        // In exact arithmetic for a symmetric matrix  -1 <= r <= 1
        // but computation error can leave it slightly outside this range.
        T phi;
        if (r <= -1)
            phi = static_cast<T>(1.047197551196598);
        else if (r >= 1)
            phi = 0;
        else
            phi = std::acos(r) / static_cast<T>(3.);

        // the eigenvalues satisfy eig3 <= eig2 <= eig1
        eigs[0] = q + static_cast<T>(2) * p * std::cos(phi);
        eigs[2] = q + static_cast<T>(2) * p * std::cos(phi + static_cast<T>(2.094395102393195));
        eigs[1] = static_cast<T>(3) * q - eigs[0] - eigs[2]; // since trace(cov) = eig1 + eig2 + eig3

        if (std::abs(eigs[0]) > std::numeric_limits<T>::epsilon()) {
            for (int index : {0, 1, 2}) {
                prod = (cov - eigs[index] * I);

                const T *col0 = prod.data();
                const T *col1 = prod.data() + 3;
                const T *col2 = prod.data() + 6;

                std::vector<T> c0xc1{col0[1] * col1[2] - col0[2] * col1[1],
                                     col0[2] * col1[0] - col0[0] * col1[2],
                                     col0[0] * col1[1] - col0[1] * col1[0]};

                std::vector<T> c0xc2 = {col0[1] * col2[2] - col0[2] * col2[1],
                                        col0[2] * col2[0] - col0[0] * col2[2],
                                        col0[0] * col2[1] - col0[1] * col2[0]};

                std::vector<T> c1xc2 = {col1[1] * col2[2] - col1[2] * col2[1],
                                        col1[2] * col2[0] - col1[0] * col2[2],
                                        col1[0] * col2[1] - col1[1] * col2[0]};

                T d0     = dprod(c0xc1, c0xc1);
                T d1     = dprod(c0xc2, c0xc2);
                T d2     = dprod(c1xc2, c1xc2);
                T dmax   = d0;
                int imax = 0;
                if (d1 > dmax) {
                    dmax = d1;
                    imax = 1;
                }
                if (d2 > dmax) {
                    imax = 2;
                }

                if (imax == 0) {
                    c0xc1 /= std::sqrt(d0);
                    result(0, index) = c0xc1[0];
                    result(1, index) = c0xc1[1];
                    result(2, index) = c0xc1[2];
                } else if (imax == 1) {
                    c0xc2 /= std::sqrt(d1);
                    result(0, index) = c0xc2[0];
                    result(1, index) = c0xc2[1];
                    result(2, index) = c0xc2[2];
                } else {
                    c1xc2 /= std::sqrt(d2);
                    result(0, index) = c1xc2[0];
                    result(1, index) = c1xc2[1];
                    result(2, index) = c1xc2[2];
                }
            }
        } else {
            result(0, 0) = 1;
            result(1, 1) = 1;
            result(2, 2) = 1;
        }
    }
    return result;
}
} // namespace htool
#endif
