#include "test_blocks.hpp"
#include <htool/clustering/ncluster.hpp>

using namespace std;
using namespace htool;

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    bool test = 0;

    test = test || (test_blocks<RegularClustering, RjasanowSteinbach>(argc, argv, false));

    MPI_Finalize();
    return test;
}