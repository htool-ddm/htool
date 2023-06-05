#include "test_hmatrix_hmatrix_product.hpp"     // for test_hmatrix_hmatrix...
#include "test_hmatrix_lrmat_product.hpp"       // for test_hmatrix_lrmat_p...
#include "test_hmatrix_matrix_product.hpp"      // for test_hermitian_hmatr...
#include "test_lrmat_hmatrix_product.hpp"       // for test_lrmat_hmatrix_p...
#include "test_matrix_hmatrix_product.hpp"      // for test_matrix_hmatrix_...
#include <array>                                // for array
#include <htool/misc/misc.hpp>                  // for underlying_type
#include <htool/testing/generate_test_case.hpp> // for TestCaseSymmetricPro...
#include <mpi.h>                                // for MPI_COMM_WORLD, MPI_...

using namespace std;
using namespace htool;

template <typename T, typename GeneratorTestType>
bool test_hmatrix_product(char transa, char transb, int n1, int n2, int n3, bool use_local_cluster, htool::underlying_type<T> epsilon, std::array<htool::underlying_type<T>, 5> additional_lrmat_sum_tolerances) {
    // Get the number of processes
    int sizeWorld;
    MPI_Comm_size(MPI_COMM_WORLD, &sizeWorld);

    // Get the rankWorld of the process
    int rankWorld;
    MPI_Comm_rank(MPI_COMM_WORLD, &rankWorld);

    bool is_error = false;
    TestCaseProduct<T, GeneratorTestType> test_case(transa, transb, n1, n2, n3, 1, 2, sizeWorld);

    is_error = is_error || test_hmatrix_matrix_product<T, GeneratorTestType>(test_case, use_local_cluster, epsilon, additional_lrmat_sum_tolerances[0]);
    is_error = is_error || test_matrix_hmatrix_product<T, GeneratorTestType>(test_case, use_local_cluster, epsilon, additional_lrmat_sum_tolerances[1]);
    is_error = is_error || test_hmatrix_lrmat_product<T, GeneratorTestType>(test_case, use_local_cluster, epsilon, additional_lrmat_sum_tolerances[2]);
    is_error = is_error || test_lrmat_hmatrix_product<T, GeneratorTestType>(test_case, use_local_cluster, epsilon, additional_lrmat_sum_tolerances[3]);
    is_error = is_error || test_hmatrix_hmatrix_product<T, GeneratorTestType>(test_case, use_local_cluster, epsilon, additional_lrmat_sum_tolerances[4]);
    return is_error;
}

template <typename T, typename GeneratorTestType>
bool test_symmetric_hmatrix_product(int n1, int n2, char side, char UPLO, htool::underlying_type<T> epsilon, htool::underlying_type<T> margin) {
    // Get the number of processes
    int sizeWorld;
    MPI_Comm_size(MPI_COMM_WORLD, &sizeWorld);

    // Get the rankWorld of the process
    int rankWorld;
    MPI_Comm_rank(MPI_COMM_WORLD, &rankWorld);

    bool is_error = false;
    TestCaseSymmetricProduct<T, GeneratorTestType> test_case(n1, n2, 2, side, 'S', UPLO, sizeWorld);

    is_error = test_symmetric_hmatrix_matrix_product<T, GeneratorTestType>(test_case, epsilon, margin);
    // is_error = test_symmetric_hmatrix_lrmat_product<T, GeneratorTestType>(test_case, use_local_cluster, epsilon);
    // is_error = test_symmetric_hmatrix_hmatrix_product<T, GeneratorTestType>(test_case, use_local_cluster, epsilon, margin);
    // if (!(side == 'L' && Symmetry != 'N')) {
    // is_error = test_lrmat_hmatrix_product<T, GeneratorTestType>(test_case, use_local_cluster, epsilon);
    // }
    return is_error;
}

template <typename T, typename GeneratorTestType>
bool test_hermitian_hmatrix_product(int n1, int n2, char side, char UPLO, htool::underlying_type<T> epsilon, htool::underlying_type<T> margin) {
    // Get the number of processes
    int sizeWorld;
    MPI_Comm_size(MPI_COMM_WORLD, &sizeWorld);

    // Get the rankWorld of the process
    int rankWorld;
    MPI_Comm_rank(MPI_COMM_WORLD, &rankWorld);

    bool is_error = false;
    TestCaseSymmetricProduct<T, GeneratorTestType> test_case(n1, n2, 2, side, 'H', UPLO, sizeWorld);

    is_error = test_hermitian_hmatrix_matrix_product<T, GeneratorTestType>(test_case, epsilon, margin);
    // is_error = test_symmetric_hmatrix_lrmat_product<T, GeneratorTestType>(test_case, use_local_cluster, epsilon);
    // is_error = test_symmetric_hmatrix_hmatrix_product<T, GeneratorTestType>(test_case, use_local_cluster, epsilon, margin);
    // if (!(side == 'L' && Symmetry != 'N')) {
    // is_error = test_lrmat_hmatrix_product<T, GeneratorTestType>(test_case, use_local_cluster, epsilon);
    // }
    return is_error;
}
