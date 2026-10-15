#include "test_local_hmatrix_solvers.hpp" // for test_local_hmatrix_solvers
#include <mpi.h>                          // for MPI_Init

// Local solvers with omp_task_policy
int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    bool is_error = test_local_hmatrix_solvers<true>();
    MPI_Finalize();
    return is_error ? 1 : 0;
}
