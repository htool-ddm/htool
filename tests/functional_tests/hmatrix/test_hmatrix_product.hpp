#include "test_hmatrix_hmatrix_product.hpp"
#include "test_hmatrix_lrmat_product.hpp"
#include "test_hmatrix_matrix_product.hpp"
#include "test_lrmat_hmatrix_product.hpp"
#include "test_matrix_hmatrix_product.hpp"
#include <htool/clustering/clustering.hpp>
#include <htool/hmatrix/hmatrix.hpp>
#include <htool/hmatrix/hmatrix_output.hpp>
#include <htool/hmatrix/linalg/interface.hpp>
#include <htool/hmatrix/tree_builder/tree_builder.hpp>
#include <htool/testing/generate_test_case.hpp>
#include <htool/testing/generator_input.hpp>
#include <htool/testing/generator_test.hpp>
#include <htool/testing/geometry.hpp>
#include <htool/testing/partition.hpp>

using namespace std;
using namespace htool;

template <typename T, typename GeneratorTestType>
bool test_hmatrix_product(char transa, char transb, int n1, int n2, int n3, char side, char Symmetry, char UPLO, bool use_local_cluster, htool::underlying_type<T> epsilon, htool::underlying_type<T> margin) {
    // Get the number of processes
    int sizeWorld;
    MPI_Comm_size(MPI_COMM_WORLD, &sizeWorld);

    // Get the rankWorld of the process
    int rankWorld;
    MPI_Comm_rank(MPI_COMM_WORLD, &rankWorld);

    bool is_error = false;
    TestCase<T, GeneratorTestType> test_case(transa, transb, n1, n2, n3, 1, 2, side, Symmetry, UPLO, sizeWorld);

    is_error = test_hmatrix_matrix_product<T, GeneratorTestType>(test_case, use_local_cluster, epsilon, margin);
    is_error = test_matrix_hmatrix_product<T, GeneratorTestType>(test_case, use_local_cluster, epsilon, margin);
    is_error = test_hmatrix_lrmat_product<T, GeneratorTestType>(test_case, use_local_cluster, epsilon);
    if (!(side == 'L' && Symmetry != 'N')) {
        is_error = test_lrmat_hmatrix_product<T, GeneratorTestType>(test_case, use_local_cluster, epsilon);
    }
    is_error = test_hmatrix_hmatrix_product<T, GeneratorTestType>(test_case, use_local_cluster, epsilon, margin);
    return is_error;
}
