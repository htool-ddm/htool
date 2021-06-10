#ifndef HTOOL_CLUSTERING_SPLITTING_HPP
#define HTOOL_CLUSTERING_SPLITTING_HPP

#include "virtual_cluster.hpp"

namespace htool {

enum class SplittingTypes { GeometricSplitting,
                            RegularSplitting };

std::vector<std::vector<int>> regular_splitting(const double *const x, const int *const tab, std::vector<int> &num, VirtualCluster const *const curr_cluster, int nb_sons, const std::vector<double> &dir) {

    std::vector<std::vector<int>> numbering(nb_sons);
    int space_dim = curr_cluster->get_space_dim();

    std::sort(num.begin(), num.end(), [&](int a, int b) {
        double c = std::inner_product(x + space_dim * tab[a], x + space_dim * (1 + tab[a]), dir.data(), double(0));
        double d = std::inner_product(x + space_dim * tab[b], x + space_dim * (1 + tab[b]), dir.data(), double(0));
        return c < d;
    });

    int size_numbering = num.size() / nb_sons;
    int count_size     = 0;
    for (int p = 0; p < nb_sons - 1; p++) {
        numbering[p].resize(size_numbering);
        std::copy_n(num.begin() + count_size, size_numbering, numbering[p].begin());
        count_size += size_numbering;
    }

    numbering.back().resize(num.size() - count_size);
    std::copy(num.begin() + count_size, num.end(), numbering.back().begin());

    return numbering;
}

std::vector<std::vector<int>> geometric_splitting(const double *const x, const int *const tab, std::vector<int> &num, VirtualCluster const *const curr_cluster, int nb_sons, const std::vector<double> &dir) {
    std::vector<std::vector<int>> numbering(nb_sons);

    // Geometry of current cluster
    int nb_pt              = curr_cluster->get_size();
    int space_dim          = curr_cluster->get_space_dim();
    std::vector<double> xc = curr_cluster->get_ctr();

    // For 2 sons, we can use the center of the cluster
    if (nb_sons == 2) {
        for (int j = 0; j < nb_pt; j++) {
            std::vector<double> dx(x + space_dim * tab[num[j]], x + space_dim * (1 + tab[num[j]]));
            for (int p = 0; p < dx.size(); p++) {
                dx[p] = dx[p] - xc[p];
            }

            if (dprod(dir, dx) > 0) {
                numbering[0].push_back(num[j]);
            } else {
                numbering[1].push_back(num[j]);
            }
        }

    }
    // Otherwise we have to something more
    else if (num.size() > 1) {
        const auto minmax = std::minmax_element(num.begin(), num.end(), [&](int a, int b) {
            double c = std::inner_product(x + space_dim * tab[a], x + space_dim * (1 + tab[a]), dir.data(), double(0));
            double d = std::inner_product(x + space_dim * tab[b], x + space_dim * (1 + tab[b]), dir.data(), double(0));
            return c < d;
        });
        std::vector<double> min(x + space_dim * tab[*(minmax.first)], x + space_dim * (1 + tab[*(minmax.first)]));
        std::vector<double> max(x + space_dim * tab[*(minmax.second)], x + space_dim * (1 + tab[*(minmax.second)]));

        double length = dprod(max - min, dir) / (double)nb_sons;
        for (int j = 0; j < nb_pt; j++) {
            std::vector<double> dx(x + space_dim * tab[num[j]], x + space_dim * (1 + tab[num[j]]));

            int index = dprod(dx - min, dir) / length;
            index     = (index == nb_sons) ? index - 1 : index; // for max
            numbering[index].push_back(num[j]);
        }

        // Check that no son is empty
        bool empty = false;
        for (int p = 0; p < numbering.size(); p++) {
            if (numbering[p].size() == 0) {
                empty = true;
            }
        }
        // In this case, we do a regular splitting
        if (empty) {
            numbering = regular_splitting(x, tab, num, curr_cluster, nb_sons, dir);
        }
    }

    return numbering;
}

} // namespace htool

#endif