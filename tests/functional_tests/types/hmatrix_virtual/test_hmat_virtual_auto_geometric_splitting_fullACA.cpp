#include "test_hmat_virtual_auto.hpp"
#include <htool/lrmat/fullACA.hpp>

int main(int argc, char *argv[]) {

    return test_hmat_virtual_auto<GeometricClustering, fullACA>(argc, argv);
}
