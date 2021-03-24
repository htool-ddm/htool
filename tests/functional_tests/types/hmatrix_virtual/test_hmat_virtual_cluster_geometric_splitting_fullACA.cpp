#include "test_hmat_virtual_cluster.hpp"
#include <htool/lrmat/fullACA.hpp>

int main(int argc, char *argv[]) {

    return test_hmat_virtual_cluster<GeometricClustering, fullACA>(argc, argv);
}
