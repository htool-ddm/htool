
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
bool test_lrmat_matrix_product(const TestCase<T, GeneratorTestType> &test_case, htool::underlying_type<T> epsilon, htool::underlying_type<T> additional_compression_tolerance, htool::underlying_type<T> additional_lrmat_sum_tolerance) {
    bool is_error = false;
    char transa   = test_case.transa;

    // ACA automatic building
    Compressor compressor;
    LowRankMatrix<T> A_auto_approximation(*test_case.operator_A, compressor, *test_case.root_cluster_A_output, *test_case.root_cluster_A_input, -1, epsilon);
    LowRankMatrix<T> C_auto_approximation(*test_case.operator_C, compressor, *test_case.root_cluster_C_output, *test_case.root_cluster_C_input, -1, epsilon);

    // partialACA fixed rank
    int reqrank_max = 10;
    LowRankMatrix<T> A_fixed_approximation(*test_case.operator_A, compressor, *test_case.root_cluster_A_output, *test_case.root_cluster_A_input, std::max(A_auto_approximation.rank_of(), reqrank_max), epsilon);
    LowRankMatrix<T> C_fixed_approximation(*test_case.operator_C, compressor, *test_case.root_cluster_C_output, *test_case.root_cluster_C_input, std::max(C_auto_approximation.rank_of(), reqrank_max), epsilon);
    LowRankMatrix<T> lrmat_test(epsilon);

    // Reference matrix
    Matrix<T> A_dense(test_case.no_A, test_case.ni_A), B_dense(test_case.no_B, test_case.ni_B), C_dense(test_case.no_C, test_case.ni_C);
    test_case.operator_A->copy_submatrix(test_case.no_A, test_case.ni_A, 0, 0, A_dense.data());
    test_case.operator_B->copy_submatrix(test_case.no_B, test_case.ni_B, 0, 0, B_dense.data());
    test_case.operator_C->copy_submatrix(test_case.no_C, test_case.ni_C, 0, 0, C_dense.data());
    Matrix<T> matrix_test, dense_lrmat_test, transposed_B_dense, transposed_C_dense;
    transpose(B_dense, transposed_B_dense);
    transpose(C_dense, transposed_C_dense);

    // Random Input
    htool::underlying_type<T> error;
    std::vector<T> B_vec, C_vec, test_vec;
    B_vec = B_dense.get_col(0);
    C_vec = C_dense.get_col(0);
    T alpha(1), beta(1), scaling_coefficient;
    generate_random_scalar(alpha);
    generate_random_scalar(beta);
    generate_random_scalar(scaling_coefficient);

    // Reference matrix
    Matrix<T> matrix_result_w_matrix_sum(C_dense), matrix_result_wo_sum(C_dense), matrix_result_w_lrmat_sum(C_dense);
    Matrix<T> transposed_matrix_result_w_sum(transposed_C_dense);
    C_auto_approximation.copy_to_dense(matrix_result_w_lrmat_sum.data());
    add_matrix_matrix_product(transa, 'N', alpha, A_dense, B_dense, beta, matrix_result_w_matrix_sum);
    add_matrix_matrix_product(transa, 'N', alpha, A_dense, B_dense, beta, matrix_result_w_lrmat_sum);
    add_matrix_matrix_product(transa, 'N', alpha, A_dense, B_dense, T(0), matrix_result_wo_sum);
    transpose(matrix_result_w_matrix_sum, transposed_matrix_result_w_sum);

    Matrix<T> scaled_matrix_result_w_matrix_sum(matrix_result_w_matrix_sum);
    scale(scaling_coefficient, scaled_matrix_result_w_matrix_sum);

    // Tests for fixed rank
    test_vec = C_vec;
    add_lrmat_vector_product(transa, alpha, A_fixed_approximation, B_vec.data(), beta, test_vec.data());
    error    = norm2(matrix_result_w_matrix_sum.get_col(0) - test_vec) / norm2(matrix_result_w_matrix_sum.get_col(0));
    is_error = is_error || !(error < epsilon * (1 + additional_compression_tolerance));
    cout << "> Errors on a lrmat vector product with fixed approximation: " << error << endl;

    matrix_test = C_dense;
    add_lrmat_matrix_product(transa, 'N', alpha, A_fixed_approximation, B_dense, beta, matrix_test);
    error    = normFrob(matrix_result_w_matrix_sum - matrix_test) / normFrob(matrix_result_w_matrix_sum);
    is_error = is_error || !(error < epsilon * (1 + additional_compression_tolerance));
    cout << "> Errors on a lrmat matrix product to matrix with fixed approximation: " << error << endl;

    lrmat_test = C_fixed_approximation;
    add_lrmat_matrix_product(transa, 'N', alpha, A_fixed_approximation, B_dense, T(0), lrmat_test);
    dense_lrmat_test.resize(lrmat_test.get_U().nb_rows(), lrmat_test.get_V().nb_cols());
    lrmat_test.copy_to_dense(dense_lrmat_test.data());
    error    = normFrob(matrix_result_wo_sum - dense_lrmat_test) / normFrob(matrix_result_wo_sum);
    is_error = is_error || !(error < epsilon * (1 + additional_compression_tolerance));
    cout << "> Errors on a lrmat matrix product to lrmat with fixed approximation and without lrmat sum: " << error << endl;

    lrmat_test = C_fixed_approximation;
    add_lrmat_matrix_product(transa, 'N', alpha, A_fixed_approximation, B_dense, beta, lrmat_test);
    dense_lrmat_test.resize(lrmat_test.get_U().nb_rows(), lrmat_test.get_V().nb_cols());
    lrmat_test.copy_to_dense(dense_lrmat_test.data());
    error    = normFrob(matrix_result_w_lrmat_sum - dense_lrmat_test) / normFrob(matrix_result_w_lrmat_sum);
    is_error = is_error || !(error < epsilon * (1 + additional_compression_tolerance + additional_lrmat_sum_tolerance));
    cout << "> Errors on a lrmat matrix product to lrmat with fixed approximation and with lrmat sum: " << error << endl;

    matrix_test = transposed_C_dense;
    add_lrmat_matrix_product_row_major(transa, 'N', alpha, A_fixed_approximation, transposed_B_dense.data(), beta, matrix_test.data(), C_dense.nb_cols());
    error    = normFrob(transposed_matrix_result_w_sum - matrix_test) / normFrob(transposed_matrix_result_w_sum);
    is_error = is_error || !(error < epsilon * (1 + additional_compression_tolerance));
    cout << "> Errors on a lrmat matrix product to matrix with fixed approximation and row major input: " << error << endl;

    matrix_test = C_dense;
    scale(scaling_coefficient, A_fixed_approximation);
    add_lrmat_matrix_product(transa, 'N', alpha, A_fixed_approximation, B_dense, scaling_coefficient * beta, matrix_test);
    error    = normFrob(scaled_matrix_result_w_matrix_sum - matrix_test) / normFrob(scaled_matrix_result_w_matrix_sum);
    is_error = is_error || !(error < epsilon * (1 + additional_compression_tolerance));
    cout << "> Errors on a scaled lrmat matrix product with fixed approximation: " << error << endl;

    // Tests for automatic rank
    test_vec = C_vec;
    add_lrmat_vector_product(transa, alpha, A_auto_approximation, B_vec.data(), beta, test_vec.data());
    error    = norm2(matrix_result_w_matrix_sum.get_col(0) - test_vec) / norm2(matrix_result_w_matrix_sum.get_col(0));
    is_error = is_error || !(error < epsilon * (1 + additional_compression_tolerance));
    cout << "> Errors on a lrmat vector product with auto approximation: " << error << endl;

    matrix_test = C_dense;
    add_lrmat_matrix_product(transa, 'N', alpha, A_auto_approximation, B_dense, beta, matrix_test);
    error    = normFrob(matrix_result_w_matrix_sum - matrix_test) / normFrob(matrix_result_w_matrix_sum);
    is_error = is_error || !(error < epsilon * (1 + additional_compression_tolerance));
    cout << "> Errors on a lrmat matrix product to matrix with auto approximation: " << error << endl;

    lrmat_test = C_auto_approximation;
    add_lrmat_matrix_product(transa, 'N', alpha, A_auto_approximation, B_dense, T(0), lrmat_test);
    dense_lrmat_test.resize(lrmat_test.get_U().nb_rows(), lrmat_test.get_V().nb_cols());
    lrmat_test.copy_to_dense(dense_lrmat_test.data());
    error    = normFrob(matrix_result_wo_sum - dense_lrmat_test) / normFrob(matrix_result_wo_sum);
    is_error = is_error || !(error < epsilon * (1 + additional_compression_tolerance));
    cout << "> Errors on a lrmat matrix product to lrmat with auto approximation and without lrmat sum: " << error << endl;

    lrmat_test = C_auto_approximation;
    add_lrmat_matrix_product(transa, 'N', alpha, A_auto_approximation, B_dense, beta, lrmat_test);
    dense_lrmat_test.resize(lrmat_test.get_U().nb_rows(), lrmat_test.get_V().nb_cols());
    lrmat_test.copy_to_dense(dense_lrmat_test.data());
    error    = normFrob(matrix_result_w_lrmat_sum - dense_lrmat_test) / normFrob(matrix_result_w_lrmat_sum);
    is_error = is_error || !(error < epsilon * (1 + additional_compression_tolerance + additional_lrmat_sum_tolerance));
    cout << "> Errors on a lrmat matrix product to lrmat with auto approximation and with lrmat sum: " << error << endl;

    matrix_test = transposed_C_dense;
    add_lrmat_matrix_product_row_major(transa, 'N', alpha, A_auto_approximation, transposed_B_dense.data(), beta, matrix_test.data(), C_dense.nb_cols());
    error    = normFrob(transposed_matrix_result_w_sum - matrix_test) / normFrob(transposed_matrix_result_w_sum);
    is_error = is_error || !(error < epsilon * (1 + additional_compression_tolerance));
    cout << "> Errors on a lrmat matrix product to matrix with auto approximation and row major input: " << error << endl;

    matrix_test = C_dense;
    scale(scaling_coefficient, A_auto_approximation);
    add_lrmat_matrix_product(transa, 'N', alpha, A_auto_approximation, B_dense, scaling_coefficient * beta, matrix_test);
    error    = normFrob(scaled_matrix_result_w_matrix_sum - matrix_test) / normFrob(scaled_matrix_result_w_matrix_sum);
    is_error = is_error || !(error < epsilon * (1 + additional_compression_tolerance));
    cout << "> Errors on a scaled lrmat matrix product with auto approximation: " << error << endl;
    cout << "test : " << is_error << endl
         << endl;

    return is_error;
}
