#include <complex>
#include <iostream>
#include <vector>

#include "../test_lrmat_build.hpp"
#include <htool/clustering/clustering.hpp>
#include <htool/hmatrix/lrmat/fullACA.hpp>
#include <htool/testing/geometry.hpp>
#include <mpi.h>

using namespace std;
using namespace htool;

int main(int argc, char *argv[]) {
    // Initialize the MPI environment
    MPI_Init(&argc, &argv);

    const int ndistance = 4;
    double distance[ndistance];
    distance[0] = 15;
    distance[1] = 20;
    distance[2] = 30;
    distance[3] = 40;

    double epsilon = 0.0001;
    int nr         = 500;
    int nc         = 100;
    std::vector<double> xt(3 * nr);
    std::vector<double> xs(3 * nc);
    std::vector<int> tabt(500);
    std::vector<int> tabs(100);
    bool test = 0;
    for (int idist = 0; idist < ndistance; idist++) {

        srand(1);
        // we set a constant seed for rand because we want always the same result if we run the check many times
        // (two different initializations with the same seed will generate the same succession of results in the subsequent calls to rand)
        create_disk(3, 0., nr, xt.data());
        create_disk(3, distance[idist], nc, xs.data());

        ClusterTreeBuilder<double, ComputeLargestExtent<double>, RegularSplitting<double>> target_recursive_build_strategy(nr, 3, xt.data(), 2, 2);
        ClusterTreeBuilder<double, ComputeLargestExtent<double>, RegularSplitting<double>> source_recursive_build_strategy(nc, 3, xs.data(), 2, 2);

        std::shared_ptr<Cluster<double>> target_cluster = std::make_shared<Cluster<double>>(target_recursive_build_strategy.create_cluster_tree());
        std::shared_ptr<Cluster<double>> source_cluster = std::make_shared<Cluster<double>>(source_recursive_build_strategy.create_cluster_tree());

        GeneratorTestDouble A(3, nr, nc, xt, xs, target_cluster, source_cluster);

        // fullACA fixed rank
        int reqrank_max = 10;
        fullACA<double> compressor;
        LowRankMatrix<double> A_fullACA_fixed(A, compressor, *target_cluster, *source_cluster, reqrank_max, epsilon);

        // ACA automatic building
        LowRankMatrix<double> A_fullACA(A, compressor, *target_cluster, *source_cluster, -1, epsilon);
        std::pair<double, double> fixed_compression_interval(0.87, 0.89);
        std::pair<double, double> auto_compression_interval(0.95, 0.97);
        test = test || (test_lrmat(*target_cluster, *source_cluster, A, A_fullACA_fixed, A_fullACA, fixed_compression_interval, auto_compression_interval));
    }
    cout << "test : " << test << endl;

    // Finalize the MPI environment.
    MPI_Finalize();
    return test;
}
