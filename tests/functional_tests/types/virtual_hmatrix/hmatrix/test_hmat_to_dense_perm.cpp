#include "test_hmat_to_dense_perm.hpp"

using namespace std;
using namespace htool;

int main(int argc, char *argv[]) {

    // Initialize the MPI environment
    MPI_Init(&argc, &argv);

    bool test = test_hmat_to_dense_perm(argc, argv);

    test = test || test_hmat_to_dense_perm_sym(argc, argv, 'U');
    test = test || test_hmat_to_dense_perm_sym(argc, argv, 'L');

    test = test || test_hmat_to_dense_perm_sym_complex(argc, argv, 'U');
    test = test || test_hmat_to_dense_perm_sym_complex(argc, argv, 'L');

    test = test || test_hmat_to_dense_perm_hermitian_complex(argc, argv, 'U');
    test = test || test_hmat_to_dense_perm_hermitian_complex(argc, argv, 'L');

    // Finalize the MPI environment.
    MPI_Finalize();

    return test;
}
