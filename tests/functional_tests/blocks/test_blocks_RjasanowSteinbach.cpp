#include "test_blocks.hpp"

using namespace std;
using namespace htool;

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    bool test = 0;

    test = test || (test_blocks<RjasanowSteinbach>(argc, argv, false));

    MPI_Finalize();
    return test;
}