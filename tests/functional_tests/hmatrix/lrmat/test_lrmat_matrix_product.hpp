#include <htool/basic_types/vector.hpp>
#include <htool/hmatrix/lrmat/linalg/add_lrmat_matrix_product.hpp>
#include <htool/hmatrix/lrmat/linalg/add_lrmat_matrix_product_row_major.hpp>
#include <htool/hmatrix/lrmat/linalg/add_lrmat_vector_product.hpp>
#include <htool/hmatrix/lrmat/linalg/scale.hpp>
#include <htool/hmatrix/lrmat/lrmat.hpp>
#include <htool/matrix/linalg/scale.hpp>
#include <htool/matrix/linalg/transpose.hpp>
#include <htool/matrix/matrix.hpp>
#include <htool/misc/misc.hpp>
#include <iostream>
#include <vector>

using namespace std;
using namespace htool;

template <typename T, typename GeneratorTestType, class Compressor>
bool test_lrmat_matrix_product(char transa, char transb, T alpha, T beta, T scaling_coefficient, const LowRankMatrix<T> &A_auto_approximation, LowRankMatrix<T> &C_auto_approximation, const Matrix<T> &B_dense, const Matrix<T> &C_dense, const Matrix<T> &matrix_result_w_matrix_sum, const Matrix<T> &matrix_result_wo_sum, const Matrix<T> &matrix_result_w_lrmat_sum, htool::underlying_type<T> epsilon, htool::underlying_type<T> additional_compression_tolerance, htool::underlying_type<T> additional_lrmat_sum_tolerance) {
    bool is_error = false;
    LowRankMatrix<T> lrmat_test(A_auto_approximation.nb_rows(), B_dense.nb_cols(), epsilon);

    // Reference matrix
    Matrix<T> matrix_test, dense_lrmat_test, transposed_B_dense(B_dense.nb_cols(), B_dense.nb_rows()), transposed_C_dense(C_dense.nb_cols(), C_dense.nb_rows());
    transpose(B_dense, transposed_B_dense);
    transpose(C_dense, transposed_C_dense);

    // Random Input
    htool::underlying_type<T> error;
    std::vector<T> B_vec, C_vec, test_vec;
    B_vec = get_col(B_dense, 0);
    C_vec = get_col(C_dense, 0);

    // Reference matrix
    Matrix<T> transposed_matrix_result_w_sum(transposed_C_dense);
    transpose(matrix_result_w_matrix_sum, transposed_matrix_result_w_sum);
    Matrix<T> scaled_matrix_result_w_matrix_sum(matrix_result_w_matrix_sum);
    scale(scaling_coefficient, scaled_matrix_result_w_matrix_sum);

    // Tests for automatic rank
    if (transb == 'N') {
        test_vec = C_vec;
        add_lrmat_vector_product(transa, alpha, A_auto_approximation, B_vec.data(), beta, test_vec.data());
        error    = norm2(get_col(matrix_result_w_matrix_sum, 0) - test_vec) / norm2(get_col(matrix_result_w_matrix_sum, 0));
        is_error = is_error || !(error < epsilon * (1 + additional_compression_tolerance));
        cout << "> Errors on a lrmat vector product with auto approximation: " << error << endl;
    }

    matrix_test = C_dense;
    add_lrmat_matrix_product(transa, transb, alpha, A_auto_approximation, B_dense, beta, matrix_test);
    error    = normFrob(matrix_result_w_matrix_sum - matrix_test) / normFrob(matrix_result_w_matrix_sum);
    is_error = is_error || !(error < epsilon * (1 + additional_compression_tolerance));
    cout << "> Errors on a lrmat matrix product to matrix with auto approximation: " << error << endl;

    lrmat_test = C_auto_approximation;
    add_lrmat_matrix_product(transa, transb, alpha, A_auto_approximation, B_dense, T(0), lrmat_test);
    dense_lrmat_test.resize(lrmat_test.get_U().nb_rows(), lrmat_test.get_V().nb_cols());
    lrmat_test.copy_to_dense(dense_lrmat_test.data());
    error    = normFrob(matrix_result_wo_sum - dense_lrmat_test) / normFrob(matrix_result_wo_sum);
    is_error = is_error || !(error < epsilon * (1 + additional_compression_tolerance));
    cout << "> Errors on a lrmat matrix product to lrmat with auto approximation and without lrmat sum: " << error << endl;

    lrmat_test = C_auto_approximation;
    add_lrmat_matrix_product(transa, transb, alpha, A_auto_approximation, B_dense, beta, lrmat_test);
    dense_lrmat_test.resize(lrmat_test.get_U().nb_rows(), lrmat_test.get_V().nb_cols());
    lrmat_test.copy_to_dense(dense_lrmat_test.data());
    error    = normFrob(matrix_result_w_lrmat_sum - dense_lrmat_test) / normFrob(matrix_result_w_lrmat_sum);
    is_error = is_error || !(error < epsilon * (1 + additional_compression_tolerance + additional_lrmat_sum_tolerance));
    cout << "> Errors on a lrmat matrix product to lrmat with auto approximation and with lrmat sum: " << error << endl;

    matrix_test = transposed_C_dense;
    add_lrmat_matrix_product_row_major(transa, transb, alpha, A_auto_approximation, transposed_B_dense.data(), beta, matrix_test.data(), C_dense.nb_cols());
    error    = normFrob(transposed_matrix_result_w_sum - matrix_test) / normFrob(transposed_matrix_result_w_sum);
    is_error = is_error || !(error < epsilon * (1 + additional_compression_tolerance));
    cout << "> Errors on a lrmat matrix product to matrix with auto approximation and row major input: " << error << endl;

    matrix_test                                  = C_dense;
    LowRankMatrix<T> scaled_A_auto_approximation = A_auto_approximation;
    scale(transa != 'C' ? scaling_coefficient : conj_if_complex(scaling_coefficient), scaled_A_auto_approximation);
    add_lrmat_matrix_product(transa, transb, alpha, scaled_A_auto_approximation, B_dense, scaling_coefficient * beta, matrix_test);
    error    = normFrob(scaled_matrix_result_w_matrix_sum - matrix_test) / normFrob(scaled_matrix_result_w_matrix_sum);
    is_error = is_error || !(error < epsilon * (1 + additional_compression_tolerance));
    cout << "> Errors on a scaled lrmat matrix product with auto approximation: " << error << endl;
    cout << "test : " << is_error << endl
         << endl;

    return is_error;
}
