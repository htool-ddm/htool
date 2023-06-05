
#ifndef HTOOL_TESTING_GEOMETRY_HPP
#define HTOOL_TESTING_GEOMETRY_HPP

#include "../basic_types/matrix.hpp"
#include <random>
#include <vector>

namespace htool {

template <typename T>
void create_disk(int space_dim, T z, int nr, T *const xt) {

    // double z1 = z;
    // for (int j = 0; j < nr; j++) {
    //     double rho            = ((double)rand() / (double)(RAND_MAX)); // (double) otherwise integer division!
    //     double theta          = ((double)rand() / (double)(RAND_MAX));
    //     xt[space_dim * j + 0] = sqrt(rho) * cos(2 * M_PI * theta);
    //     xt[space_dim * j + 1] = sqrt(rho) * sin(2 * M_PI * theta);
    //     if (space_dim == 3)
    //         xt[space_dim * j + 2] = z1;
    //     // sqrt(rho) otherwise the points would be concentrated in the center of the disk
    // }

    // std::random_device rd;
    std::mt19937 mersenne_engine(0);
    std::uniform_real_distribution<T> dist(0, 1);
    auto gen = [&dist, &mersenne_engine]() {
        return dist(mersenne_engine);
    };
    T z1 = z;
    for (int j = 0; j < nr; j++) {
        T rho                 = gen(); // (T) otherwise integer division!
        T theta               = gen();
        xt[space_dim * j + 0] = sqrt(rho) * cos(2 * static_cast<T>(M_PI) * theta);
        xt[space_dim * j + 1] = sqrt(rho) * sin(2 * static_cast<T>(M_PI) * theta);
        if (space_dim == 3)
            xt[space_dim * j + 2] = z1;
        // sqrt(rho) otherwise the points would be concentrated in the center of the disk
    }
}
} // namespace htool

#endif
