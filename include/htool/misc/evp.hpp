#ifndef HTOOL_EVP_HPP
#define HTOOL_EVP_HPP

#include "../types/matrix.hpp"
#include <iostream>
#include <vector>

namespace htool {

inline std::vector<double> solve_EVP_2(const Matrix<double> &cov) {
    std::vector<double> dir(2, 0);
    std::vector<double> eigs(2);
    Matrix<double> I(2, 2);
    I(0, 0) = 1;
    I(1, 1) = 1;
    Matrix<double> prod(2, 2);
    double trace = cov(0, 0) + cov(1, 1);
    double det   = cov(0, 0) * cov(1, 1) - cov(0, 1) * cov(1, 0);
    eigs[0]      = trace / 2. + std::sqrt((trace * trace / 4. - det));
    eigs[1]      = trace / 2. - std::sqrt((trace * trace / 4. - det));
    if (std::abs(eigs[0]) > 1e-16) {

        prod           = (cov - eigs[1] * I);
        int ind        = 0;
        double dirnorm = 0;
        do {
            dir[0]  = prod(0, ind);
            dir[1]  = prod(1, ind);
            dirnorm = sqrt(dir[0] * dir[0] + dir[1] * dir[1]);
            ind++;
        } while ((dirnorm < 1.e-15) && (ind < 2));
        if (dirnorm < 1.e-15) {
            dir[0] = 1;
            dir[1] = 0;
        } else {
            dir[0] /= dirnorm;
            dir[1] /= dirnorm;
        }
    }
    return dir;
}

inline std::vector<double> solve_EVP_3(const Matrix<double> &cov) {
    std::vector<double> dir(3, 0);
    double p1 = pow(cov(0, 1), 2) + pow(cov(0, 2), 2) + pow(cov(1, 2), 2);
    std::vector<double> eigs(3);
    Matrix<double> I(3, 3);
    I(0, 0) = 1;
    I(1, 1) = 1;
    I(2, 2) = 1;
    Matrix<double> prod(3, 3);
    if (p1 < 1e-16) {
        // cov is diagonal.
        eigs[0] = cov(0, 0);
        eigs[1] = cov(1, 1);
        eigs[2] = cov(2, 2);
        dir[0]  = 1;
        dir[1]  = 0;
        dir[2]  = 0;
        if (eigs[0] < eigs[1]) {
            double tmp = eigs[0];
            eigs[0]    = eigs[1];
            eigs[1]    = tmp;
            dir[0]     = 0;
            dir[1]     = 1;
            dir[2]     = 0;
        }
        if (eigs[0] < eigs[2]) {
            double tmp = eigs[0];
            eigs[0]    = eigs[2];
            eigs[2]    = tmp;
            dir[0]     = 0;
            dir[1]     = 0;
            dir[2]     = 1;
        }
    } else {
        double q  = (cov(0, 0) + cov(1, 1) + cov(2, 2)) / 3.;
        double p2 = pow(cov(0, 0) - q, 2) + pow(cov(1, 1) - q, 2) + pow(cov(2, 2) - q, 2) + 2. * p1;
        double p  = sqrt(p2 / 6.);
        Matrix<double> B(3, 3);
        B           = (1. / p) * (cov - q * I);
        double detB = B(0, 0) * (B(1, 1) * B(2, 2) - B(1, 2) * B(2, 1))
                      - B(0, 1) * (B(1, 0) * B(2, 2) - B(1, 2) * B(2, 0))
                      + B(0, 2) * (B(1, 0) * B(2, 1) - B(1, 1) * B(2, 0));
        double r = detB / 2.;

        // In exact arithmetic for a symmetric matrix  -1 <= r <= 1
        // but computation error can leave it slightly outside this range.
        double phi;
        if (r <= -1)
            phi = 3.14159265358979323846 / 3.;
        else if (r >= 1)
            phi = 0;
        else
            phi = acos(r) / 3.;

        // the eigenvalues satisfy eig3 <= eig2 <= eig1
        eigs[0] = q + 2. * p * cos(phi);
        eigs[2] = q + 2. * p * cos(phi + (2. * 3.14159265358979323846 / 3.));
        eigs[1] = 3. * q - eigs[0] - eigs[2]; // since trace(cov) = eig1 + eig2 + eig3

        // clean up near zero values to zeros (needed when cov has a kernel)
        if (std::abs(eigs[1]) < 1.e-12)
            eigs[1] = 0.;
        if (std::abs(eigs[2]) < 1.e-12)
            eigs[2] = 0.;

        if (std::abs(eigs[0]) < 1.e-16)
            dir *= 0.;
        else {
            prod           = (cov - eigs[1] * I) * (cov - eigs[2] * I);
            int ind        = 0;
            double dirnorm = 0;
            do {
                dir[0]  = prod(0, ind);
                dir[1]  = prod(1, ind);
                dir[2]  = prod(2, ind);
                dirnorm = sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
                ind++;
            } while ((dirnorm < 1.e-15) && (ind < 3));
            if (dirnorm < 1.e-15) {
                dir[0] = 1;
                dir[1] = 0;
                dir[2] = 0;
            } else {
                dir[0] /= dirnorm;
                dir[1] /= dirnorm;
                dir[2] /= dirnorm;
            }
        }
    }
    return dir;
}
} // namespace htool
#endif
