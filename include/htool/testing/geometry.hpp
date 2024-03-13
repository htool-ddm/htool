
#ifndef HTOOL_TESTING_GEOMETRY_HPP
#define HTOOL_TESTING_GEOMETRY_HPP

#include "../matrix/matrix.hpp"
#include <array>
#include <random>
#include <vector>
namespace htool {

template <typename T>
void create_disk(int space_dim, T z, int nr, T *const xt) {

    std::mt19937 mersenne_engine(0);
    std::uniform_real_distribution<T> dist(0, 1);
    auto gen = [&dist, &mersenne_engine]() {
        return dist(mersenne_engine);
    };
    T z1 = z;
    for (int j = 0; j < nr; j++) {
        T rho                 = gen(); // (T) otherwise integer division!
        T theta               = gen();
        xt[space_dim * j + 0] = std::sqrt(rho) * std::cos(2 * static_cast<T>(M_PI) * theta);
        xt[space_dim * j + 1] = std::sqrt(rho) * std::sin(2 * static_cast<T>(M_PI) * theta);
        if (space_dim == 3)
            xt[space_dim * j + 2] = z1;
        // sqrt(rho) otherwise the points would be concentrated in the center of the disk
    }
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
