#include "test_hmat_cluster.hpp"

int main(int argc, char *argv[]) {

    return test_hmat_cluster<GeometricClustering, SVD>(argc, argv);
}
