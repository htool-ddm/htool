#include "test_solver_ddm.hpp"

int main(int argc, char *argv[]) {

    // Initialize the MPI environment
    MPI_Init(&argc, &argv);

    bool test = test_solver_ddm(argc, argv, 10, 'N', false);

    test = test || test_solver_ddm(argc, argv, 10, 'N', true);

    // Finalize the MPI environment.
    MPI_Finalize();
    return test;
}
