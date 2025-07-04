#include "../test_lrmat_build.hpp"                        // for test_lrmat
#include <algorithm>                                      // for max, copy
#include <cmath>                                          // for pow, sqrt
#include <htool/clustering/cluster_node.hpp>              // for Cluster
#include <htool/clustering/tree_builder/tree_builder.hpp> // for Cluster...
#include <htool/hmatrix/interfaces/virtual_generator.hpp> // for Generat...
#include <htool/hmatrix/lrmat/lrmat.hpp>                  // for LowRank...
#include <htool/hmatrix/lrmat/partialACA.hpp>             // for partialACA
#include <htool/testing/generator_test.hpp>               // for Generat...
#include <htool/testing/geometry.hpp>                     // for create_...
#include <iostream>                                       // for basic_o...
#include <utility>                                        // for pair
#include <vector>                                         // for vector

using namespace std;
using namespace htool;

int main(int, char *[]) {

    const int ndistance = 4;
    double distance[ndistance];
    distance[0] = 15;
    distance[1] = 20;
    distance[2] = 30;
    distance[3] = 40;

    double epsilon = 0.0001;

    int nr = 500;
    int nc = 100;
    std::vector<double> xt(3 * nr);
    std::vector<double> xs(3 * nc);
    std::vector<int> tabt(500);
    std::vector<int> tabs(100);
    bool test = 0;
    for (int idist = 0; idist < ndistance; idist++) {

        create_disk(3, 0., nr, xt.data());
        create_disk(3, distance[idist], nc, xs.data());

        ClusterTreeBuilder<double> recursive_build_strategy;

        Cluster<double> t = recursive_build_strategy.create_cluster_tree(nr, 3, xt.data(), 2, 2);
        Cluster<double> s = recursive_build_strategy.create_cluster_tree(nc, 3, xt.data(), 2, 2);

        GeneratorTestDouble A_in_user_numbering(3, xt, xs);
        InternalGeneratorWithPermutation<double> A(A_in_user_numbering, t.get_permutation().data(), s.get_permutation().data());

        // partialACA fixed rank
        int reqrank_max = 10;
        partialACA<double> compressor(A_in_user_numbering, t.get_permutation().data(), s.get_permutation().data());

        LowRankMatrix<double> A_partialACA_fixed(t.get_size(), s.get_size(), reqrank_max, epsilon);
        compressor.copy_low_rank_approximation(t.get_size(), s.get_size(), t.get_offset(), s.get_offset(), reqrank_max, A_partialACA_fixed);

        // ACA automatic building
        LowRankMatrix<double> A_partialACA(t.get_size(), s.get_size(), epsilon);
        compressor.copy_low_rank_approximation(t.get_size(), s.get_size(), t.get_offset(), s.get_offset(), A_partialACA);

        std::pair<double, double> fixed_compression_interval(0.87, 0.89);
        std::pair<double, double> auto_compression_interval(0.93, 0.96);
        test = test || (test_lrmat(t, s, A, A_partialACA_fixed, A_partialACA, fixed_compression_interval, auto_compression_interval));
    }

    cout << "test : " << test << endl;

    return test;
}
