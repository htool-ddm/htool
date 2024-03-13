
#include <htool/clustering/clustering.hpp>
#include <htool/hmatrix/lrmat/linalg/interface.hpp>
#include <htool/hmatrix/lrmat/linalg/scale.hpp>
#include <htool/hmatrix/lrmat/lrmat.hpp>
#include <htool/testing/generator_input.hpp>
#include <htool/testing/generator_test.hpp>
#include <htool/testing/geometry.hpp>

using namespace std;
using namespace htool;

template <typename T, typename GeneratorTestType, class Compressor>
bool test_matrix_matrix_product(const TestCase<T, GeneratorTestType> &test_case, htool::underlying_type<T> epsilon, htool::underlying_type<T> additional_compression_tolerance, htool::underlying_type<T> additional_lrmat_sum_tolerance) {
    bool is_error = false;

    char transa = test_case.transa;

    // ACA automatic building
    Compressor compressor;
    LowRankMatrix<T> C_auto_approximation(*test_case.operator_C, compressor, *test_case.root_cluster_C_output, *test_case.root_cluster_C_input, -1, epsilon), lrmat_test(epsilon);

    // Dense Matrices
    Matrix<T> matrix_test, dense_lrmat_test;

    // Random Input
    htool::underlying_type<T> error;
    T alpha(1), beta(1);
    generate_random_scalar(alpha);
    generate_random_scalar(beta);

    // Reference matrix
    Matrix<T> A_dense(test_case.no_A, test_case.ni_A), B_dense(test_case.no_B, test_case.ni_B), C_dense(test_case.no_C, test_case.ni_C);
    test_case.operator_A->copy_submatrix(test_case.no_A, test_case.ni_A, 0, 0, A_dense.data());
    test_case.operator_B->copy_submatrix(test_case.no_B, test_case.ni_B, 0, 0, B_dense.data());
    test_case.operator_C->copy_submatrix(test_case.no_C, test_case.ni_C, 0, 0, C_dense.data());
    Matrix<T> matrix_result_wo_sum(C_dense), matrix_result_w_lrmat_sum(C_dense);
    C_auto_approximation.copy_to_dense(matrix_result_w_lrmat_sum.data());
    add_matrix_matrix_product(transa, 'N', alpha, A_dense, B_dense, beta, matrix_result_w_lrmat_sum);
    add_matrix_matrix_product(transa, 'N', alpha, A_dense, B_dense, T(0), matrix_result_wo_sum);

    // Product
    lrmat_test = C_auto_approximation;
    add_matrix_matrix_product(transa, 'N', alpha, A_dense, B_dense, T(0), lrmat_test);
    dense_lrmat_test.resize(lrmat_test.get_U().nb_rows(), lrmat_test.get_V().nb_cols());
    lrmat_test.copy_to_dense(dense_lrmat_test.data());
    error    = normFrob(matrix_result_wo_sum - dense_lrmat_test) / normFrob(matrix_result_wo_sum);
    is_error = is_error || !(error < epsilon);
    cout << "> Errors on a matrix matrix product to lrmat without sum: " << error << " vs " << epsilon << endl;

    lrmat_test = C_auto_approximation;
    add_matrix_matrix_product(transa, 'N', alpha, A_dense, B_dense, beta, lrmat_test);
    dense_lrmat_test.resize(lrmat_test.get_U().nb_rows(), lrmat_test.get_V().nb_cols());
    lrmat_test.copy_to_dense(dense_lrmat_test.data());
    error    = normFrob(matrix_result_w_lrmat_sum - dense_lrmat_test) / normFrob(matrix_result_w_lrmat_sum);
    is_error = is_error || !(error < epsilon * (1 + additional_compression_tolerance + additional_lrmat_sum_tolerance));
    cout << "> Errors on a matrix matrix product to lrmat with sum: " << error << " vs " << epsilon * (1 + additional_compression_tolerance + additional_lrmat_sum_tolerance) << endl;

    cout << "test : " << is_error << endl
         << endl;

    return is_error;
}
