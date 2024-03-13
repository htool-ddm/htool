
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
bool test_matrix_lrmat_product(const TestCase<T, GeneratorTestType> &test_case, htool::underlying_type<T> epsilon, htool::underlying_type<T> additional_compression_tolerance, htool::underlying_type<T> additional_lrmat_sum_tolerance) {
    bool is_error = false;
    char transa   = test_case.transa;

    // ACA automatic building
    Compressor compressor;
    LowRankMatrix<T> B_auto_approximation(*test_case.operator_B, compressor, *test_case.root_cluster_B_output, *test_case.root_cluster_B_input, -1, epsilon);
    LowRankMatrix<T> C_auto_approximation(*test_case.operator_C, compressor, *test_case.root_cluster_C_output, *test_case.root_cluster_C_input, -1, epsilon);

    // partialACA fixed rank
    int reqrank_max = 10;
    LowRankMatrix<T> B_fixed_approximation(*test_case.operator_B, compressor, *test_case.root_cluster_B_output, *test_case.root_cluster_B_input, std::max(B_auto_approximation.rank_of(), reqrank_max), epsilon);
    LowRankMatrix<T> C_fixed_approximation(*test_case.operator_C, compressor, *test_case.root_cluster_C_output, *test_case.root_cluster_C_input, std::max(C_auto_approximation.rank_of(), reqrank_max), epsilon);
    LowRankMatrix<T> lrmat_test(epsilon);

    // Random input matrix
    T alpha, beta;
    htool::underlying_type<T> error;
    generate_random_scalar(alpha);
    generate_random_scalar(beta);

    // Reference matrix
    Matrix<T> A_dense(test_case.no_A, test_case.ni_A), B_dense(test_case.no_B, test_case.ni_B), C_dense(test_case.no_C, test_case.ni_C);
    test_case.operator_A->copy_submatrix(test_case.no_A, test_case.ni_A, 0, 0, A_dense.data());
    test_case.operator_B->copy_submatrix(test_case.no_B, test_case.ni_B, 0, 0, B_dense.data());
    test_case.operator_C->copy_submatrix(test_case.no_C, test_case.ni_C, 0, 0, C_dense.data());
    Matrix<T> matrix_result_w_matrix_sum(C_dense), matrix_result_wo_sum(C_dense), matrix_test, dense_lrmat_test, matrix_result_w_lrmat_sum(C_dense);
    C_auto_approximation.copy_to_dense(matrix_result_w_lrmat_sum.data());
    add_matrix_matrix_product(transa, 'N', alpha, A_dense, B_dense, beta, matrix_result_w_matrix_sum);
    add_matrix_matrix_product(transa, 'N', alpha, A_dense, B_dense, beta, matrix_result_w_lrmat_sum);
    add_matrix_matrix_product(transa, 'N', alpha, A_dense, B_dense, T(0), matrix_result_wo_sum);

    // Products with fixed approximation
    matrix_test = C_dense;
    add_matrix_lrmat_product(transa, 'N', alpha, A_dense, B_fixed_approximation, beta, matrix_test);
    error    = normFrob(matrix_result_w_matrix_sum - matrix_test) / normFrob(matrix_result_w_matrix_sum);
    is_error = is_error || !(error < epsilon);
    cout << "> Errors on a matrix lrmat product to matrix with fixed approximation: " << error << endl;

    lrmat_test = C_fixed_approximation;
    add_matrix_lrmat_product(transa, 'N', alpha, A_dense, B_fixed_approximation, T(0), lrmat_test);
    dense_lrmat_test.resize(lrmat_test.get_U().nb_rows(), lrmat_test.get_V().nb_cols());
    lrmat_test.copy_to_dense(dense_lrmat_test.data());
    error    = normFrob(matrix_result_wo_sum - dense_lrmat_test) / normFrob(matrix_result_wo_sum);
    is_error = is_error || !(error < epsilon);
    cout << "> Errors on a matrix lrmat product to lrmat with fixed approximation and without lrmat sum: " << error << endl;

    lrmat_test = C_fixed_approximation;
    add_matrix_lrmat_product(transa, 'N', alpha, A_dense, B_fixed_approximation, beta, lrmat_test);
    dense_lrmat_test.resize(lrmat_test.get_U().nb_rows(), lrmat_test.get_V().nb_cols());
    lrmat_test.copy_to_dense(dense_lrmat_test.data());
    error    = normFrob(matrix_result_w_lrmat_sum - dense_lrmat_test) / normFrob(matrix_result_w_lrmat_sum);
    is_error = is_error || !(error < epsilon * (1 + additional_compression_tolerance + additional_lrmat_sum_tolerance));
    cout << "> Errors on a matrix lrmat product to lrmat with fixed approximation and with lrmat sum: " << error << endl;

    // Products with auto approximation
    matrix_test = C_dense;
    add_matrix_lrmat_product(transa, 'N', alpha, A_dense, B_auto_approximation, beta, matrix_test);
    error    = normFrob(matrix_result_w_matrix_sum - matrix_test) / normFrob(matrix_result_w_matrix_sum);
    is_error = is_error || !(error < epsilon);
    cout << "> Errors on a matrix lrmat product to matrix with automatic approximation: " << error << endl;

    lrmat_test = C_auto_approximation;
    add_matrix_lrmat_product(transa, 'N', alpha, A_dense, B_auto_approximation, T(0), lrmat_test);
    dense_lrmat_test.resize(lrmat_test.get_U().nb_rows(), lrmat_test.get_V().nb_cols());
    lrmat_test.copy_to_dense(dense_lrmat_test.data());
    error    = normFrob(matrix_result_wo_sum - dense_lrmat_test) / normFrob(matrix_result_wo_sum);
    is_error = is_error || !(error < epsilon);
    cout << "> Errors on a matrix lrmat product to lrmat with automatic approximation and without lrmat sum: " << error << endl;

    lrmat_test = C_auto_approximation;
    add_matrix_lrmat_product(transa, 'N', alpha, A_dense, B_auto_approximation, beta, lrmat_test);
    dense_lrmat_test.resize(lrmat_test.get_U().nb_rows(), lrmat_test.get_V().nb_cols());
    lrmat_test.copy_to_dense(dense_lrmat_test.data());
    error    = normFrob(matrix_result_w_lrmat_sum - dense_lrmat_test) / normFrob(matrix_result_w_lrmat_sum);
    is_error = is_error || !(error < epsilon * (1 + additional_compression_tolerance + additional_lrmat_sum_tolerance));
    cout << "> Errors on a matrix lrmat product to lrmat with automatic approximation and with lrmat sum: " << error << endl;

    cout << "test : " << is_error << endl
         << endl;

    return is_error;
}
