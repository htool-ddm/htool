#include "test_cluster_global.hpp"
#include "test_cluster_local.hpp"
#include <htool/clustering/ncluster.hpp>

using namespace std;
using namespace htool;

int main(int argc, char *argv[]) {

    MPI_Init(&argc, &argv);

    bool test = test_cluster_global<GeometricClustering, 2>(argc, argv);

    test = test || test_cluster_global<GeometricClustering, 3>(argc, argv);

    test = test || test_cluster_local<GeometricClustering, 2>(argc, argv);

    test = test || test_cluster_local<GeometricClustering, 3>(argc, argv);

    MPI_Finalize();
    return test;
}