#ifndef HTOOL_TESTING_SINGLE_LAYER_SPHERE_HPP
#define HTOOL_TESTING_SINGLE_LAYER_SPHERE_HPP

#include "../hmatrix/interfaces/virtual_generator.hpp" // for VirtualGenerator
#include <cmath>                                       // for sqrt, cos, sin
#include <complex>                                     // for complex, exp
#include <vector>                                      // for vector

namespace htool {

/**
 * @brief Coordinates (x, y, z of each point, contiguously) of n quasi-uniform points on the unit sphere (Fibonacci lattice).
 */
inline std::vector<double> fibonacci_sphere(int n) {
    std::vector<double> points(3 * n);
    double phi = (1 + std::sqrt(5.)) / 2;
    for (int i = 1; i <= n; i++) {
        double z = 1 - (2. * i - 1) / n, r = std::sqrt(1 - z * z), theta = 2 * M_PI * i / phi;
        points[3 * (i - 1)]     = r * std::cos(theta);
        points[3 * (i - 1) + 1] = r * std::sin(theta);
        points[3 * (i - 1) + 2] = z;
    }
    return points;
}

/**
 * @brief Nyström-like discretization of the Laplace (wavenumber 0, real T) or Helmholtz (complex T) single-layer operator on the unit sphere,
 * at the points given by fibonacci_sphere, with weights w = 4π/n.
 *
 * The diagonal is the integral of the static part of the kernel over a disk of area w centered at the point. This gives a matrix
 * representative of a first-kind boundary integral equation, whose LU factorization depends on its Schur complements (unlike a kernel
 * regularized with a small constant, which makes the matrix strongly diagonally dominant). It is single_layer_sphere of the tests of
 * HMatrices.jl.
 */
template <typename T>
class SingleLayerSphere : public VirtualGenerator<T> {
    const std::vector<double> &m_points;
    double m_weight, m_radius, m_wavenumber;

    double diagonal(double) const { return m_radius / 2; }
    std::complex<double> diagonal(std::complex<double>) const { return {m_radius / 2, m_wavenumber * m_weight / (4 * M_PI)}; }
    double kernel(double distance, double) const { return m_weight / (4 * M_PI * distance); }
    std::complex<double> kernel(double distance, std::complex<double>) const { return m_weight * std::exp(std::complex<double>(0, m_wavenumber * distance)) / (4 * M_PI * distance); }

  public:
    SingleLayerSphere(const std::vector<double> &points, double wavenumber) : m_points(points), m_weight(4 * M_PI / static_cast<double>(points.size() / 3)), m_radius(std::sqrt(m_weight / M_PI)), m_wavenumber(wavenumber) {}

    T coefficient(int i, int j) const {
        if (i == j) {
            return diagonal(T(0));
        }
        double dx = m_points[3 * i] - m_points[3 * j], dy = m_points[3 * i + 1] - m_points[3 * j + 1], dz = m_points[3 * i + 2] - m_points[3 * j + 2];
        return kernel(std::sqrt(dx * dx + dy * dy + dz * dz), T(0));
    }

    void copy_submatrix(int M, int N, const int *rows, const int *cols, T *ptr) const override {
        for (int k = 0; k < N; k++) {
            for (int j = 0; j < M; j++) {
                ptr[j + M * k] = coefficient(rows[j], cols[k]);
            }
        }
    }
};

} // namespace htool

#endif
