
#ifndef HTOOL_TESTING_GEOMETRY_HPP
#define HTOOL_TESTING_GEOMETRY_HPP

#include "../types/matrix.hpp"
#include <vector>

namespace htool {

void create_disk(int space_dim, double z, int nr, double *const xt) {

    double z1 = z;
    for (int j = 0; j < nr; j++) {
        double rho            = ((double)rand() / (double)(RAND_MAX)); // (double) otherwise integer division!
        double theta          = ((double)rand() / (double)(RAND_MAX));
        xt[space_dim * j + 0] = sqrt(rho) * cos(2 * M_PI * theta);
        xt[space_dim * j + 1] = sqrt(rho) * sin(2 * M_PI * theta);
        if (space_dim == 3)
            xt[space_dim * j + 2] = z1;
        // sqrt(rho) otherwise the points would be concentrated in the center of the disk
    }
}
} // namespace htool

#endif