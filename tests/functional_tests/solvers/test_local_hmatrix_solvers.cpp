#include "test_local_hmatrix_solvers.hpp" // for test_local_hmatrix_solvers
#include <mpi.h>                          // for MPI_Init

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    bool is_error = test_local_hmatrix_solvers<false>();
    MPI_Finalize();
    return is_error ? 1 : 0;
}
