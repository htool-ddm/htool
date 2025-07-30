#include <fstream>                                           // for basic_o...
#include <htool/hmatrix/hmatrix.hpp>                         // for HMatrix
#include <htool/hmatrix/linalg/add_lrmat_hmatrix.hpp>        // for add_lrm...
#include <htool/hmatrix/lrmat/lrmat.hpp>                     // for LowRank...
#include <htool/hmatrix/tree_builder/tree_builder.hpp>       // for HMatrix...
#include <htool/matrix/linalg/add_matrix_matrix_product.hpp> // for add_mat...
#include <htool/matrix/matrix.hpp>                           // for Matrix
#include <htool/misc/misc.hpp>                               // for underly...
#include <htool/testing/generate_test_case.hpp>              // for TestCas...
#include <iostream>                                          // for cout

using namespace std;
using namespace htool;

template <typename T, typename GeneratorTestType, class Compressor>
bool test_lrmat_hmatrix_addition(int n1, int n2, htool::underlying_type<T> epsilon, htool::underlying_type<T>) {

    bool is_error = false;
    htool::underlying_type<T> error;
    htool::underlying_type<T> eta = 10;
    TestCaseProduct<T, GeneratorTestType> test_case('N', 'N', n1, n2, n2, 1, 4);

    // HMatrix
    HMatrixTreeBuilder<T, htool::underlying_type<T>> hmatrix_tree_builder_A(epsilon, eta, 'N', 'N');
    HMatrix<T, htool::underlying_type<T>> A = hmatrix_tree_builder_A.build(*test_case.operator_A, *test_case.root_cluster_A_output, *test_case.root_cluster_A_input);
    HMatrix<T, htool::underlying_type<T>> hmatrix_test(A);

    // lrmat
    Compressor compressor(*test_case.operator_C);
    LowRankMatrix<T> C_approximation(test_case.root_cluster_C_output->get_size(), test_case.root_cluster_C_input->get_size(), epsilon);
    compressor.copy_low_rank_approximation(test_case.root_cluster_C_output->get_size(), test_case.root_cluster_C_input->get_size(), test_case.root_cluster_C_output->get_offset(), test_case.root_cluster_C_input->get_offset(), C_approximation);

    // Reference
    Matrix<T> A_dense(A.nb_rows(), A.nb_cols(), 0);
    copy_to_dense(A, A_dense.data());
    Matrix<T> C_dense(C_approximation.nb_rows(), C_approximation.nb_cols(), 0);

    Matrix<T> result(A_dense), matrix_test, densified_hmatrix_test(n1, n2);
    add_matrix_matrix_product('N', 'N', T(1), C_approximation.get_U(), C_approximation.get_V(), T(1), result);

    //
    hmatrix_test = A;
    internal_add_lrmat_hmatrix(C_approximation, hmatrix_test);
    copy_to_dense(hmatrix_test, densified_hmatrix_test.data());
    error    = normFrob(result - densified_hmatrix_test) / normFrob(result);
    is_error = is_error || !(error < epsilon);
    cout << "> Errors on a lrmat hmatrix addition: " << error << endl;

    std::ofstream densified_hmatrix_test_file("densified_hmatrix_test");
    print(densified_hmatrix_test, densified_hmatrix_test_file, ",");

    std::ofstream result_file("result");
    print(result, result_file, ",");
    return is_error;
}
