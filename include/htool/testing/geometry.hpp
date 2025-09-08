
#ifndef HTOOL_TESTING_GEOMETRY_HPP
#define HTOOL_TESTING_GEOMETRY_HPP

#include <array>  // for array
#include <cmath>  // for M_PI, cbrt
#include <random> // for mt19937, uniform_real_distribution
namespace htool {

template <typename T>
void create_rotated_ellipse(int space_dim, T a, T b, T alpha, T z, int nr, T *const xt) {
    std::mt19937 mersenne_engine(0);
    std::uniform_real_distribution<T> dist(0, 1);
    auto gen = [&dist, &mersenne_engine]() {
        return dist(mersenne_engine);
    };

    T cos_alpha = std::cos(alpha);
    T sin_alpha = std::sin(alpha);

    for (int j = 0; j < nr; j++) {
        T rho   = gen();
        T theta = gen();
        T r     = std::sqrt(rho);
        T phi   = 2 * static_cast<T>(M_PI) * theta;

        // Axis-aligned ellipse coordinates
        T x_prime = a * r * std::cos(phi);
        T y_prime = b * r * std::sin(phi);

        // Apply rotation
        xt[space_dim * j + 0] = cos_alpha * x_prime - sin_alpha * y_prime;
        xt[space_dim * j + 1] = sin_alpha * x_prime + cos_alpha * y_prime;

        if (space_dim == 3)
            xt[space_dim * j + 2] = z;
    }
}

template <typename T>
void create_disk(int space_dim, T z, int nr, T *const xt) {
    create_rotated_ellipse(space_dim, T(1.), T(1.), T(0.), z, nr, xt);
}

template <typename T>
void create_sphere(int nr, T *const xt, std::array<T, 3> offset = {0, 0, 0}) {

    std::mt19937 mersenne_engine(0);
    std::uniform_real_distribution<T> dist(0, 1);
    auto gen = [&dist, &mersenne_engine]() {
        return dist(mersenne_engine);
    };
    for (int j = 0; j < nr; j++) {
        T rho         = gen(); // (T) otherwise integer division!
        T theta       = 2 * M_PI * gen();
        T phi         = std::acos(2 * gen() - 1);
        xt[3 * j + 0] = offset[0] + std::cbrt(rho) * std::sin(phi) * std::cos(theta);
        xt[3 * j + 1] = offset[1] + std::cbrt(rho) * std::sin(phi) * std::sin(theta);
        xt[3 * j + 2] = offset[2] + std::cbrt(rho) * std::cos(phi);
    }
}

} // namespace htool

#endif
