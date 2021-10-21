#include "test_hmat_local_get.hpp"

using namespace std;
using namespace htool;

int main(int argc, char *argv[]) {
    // Initialize the MPI environment
    MPI_Init(&argc, &argv);

    bool test = test_hmat_local_get(argc, argv, 'N', 'N');

    test = test || test_hmat_local_get(argc, argv, 'S', 'U');
    test = test || test_hmat_local_get(argc, argv, 'S', 'L');

    test = test || test_hmat_local_get_complex(argc, argv, 'S', 'U');
    test = test || test_hmat_local_get_complex(argc, argv, 'S', 'L');

    test = test || test_hmat_local_get_complex_hermitian(argc, argv, 'H', 'U');
    test = test || test_hmat_local_get_complex_hermitian(argc, argv, 'H', 'L');
    // Finalize the MPI environment.
    MPI_Finalize();
    return test;
}
