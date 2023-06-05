#include <htool/hmatrix/lrmat/linalg/add_lrmat_lrmat_product.hpp>
#include <htool/hmatrix/lrmat/lrmat.hpp>
#include <htool/matrix/matrix.hpp>
#include <htool/misc/misc.hpp>
#include <iostream>

using namespace std;
using namespace htool;

template <typename T, typename GeneratorTestType, class Compressor>
bool test_lrmat_lrmat_product(char transa, char transb, T alpha, T beta, const LowRankMatrix<T> &A_auto_approximation, const LowRankMatrix<T> &B_auto_approximation, const LowRankMatrix<T> &C_auto_approximation, const Matrix<T> &C_dense, const Matrix<T> &matrix_result_w_matrix_sum, const Matrix<T> &matrix_result_wo_sum, const Matrix<T> &matrix_result_w_lrmat_sum, htool::underlying_type<T> epsilon, htool::underlying_type<T> additional_compression_tolerance, htool::underlying_type<T> additional_lrmat_sum_tolerance) {

    bool is_error = false;
    Matrix<T> matrix_test, dense_lrmat_test;
    LowRankMatrix<T> lrmat_test(epsilon);
    htool::underlying_type<T> error;

    // Product with automatic rank
    matrix_test = C_dense;
    add_lrmat_lrmat_product(transa, transb, alpha, A_auto_approximation, B_auto_approximation, beta, matrix_test);
    error    = normFrob(matrix_result_w_matrix_sum - matrix_test) / normFrob(matrix_result_w_matrix_sum);
    is_error = is_error || !(error < A_auto_approximation.get_epsilon() * (1 + additional_compression_tolerance));
    cout << "> Errors on a lrmat lrmat product to matrix with auto approximation: " << error << endl;

    lrmat_test = C_auto_approximation;
    add_lrmat_lrmat_product(transa, transb, alpha, A_auto_approximation, B_auto_approximation, T(0), lrmat_test);
    dense_lrmat_test.resize(lrmat_test.get_U().nb_rows(), lrmat_test.get_V().nb_cols());
    lrmat_test.copy_to_dense(dense_lrmat_test.data());
    error    = normFrob(matrix_result_wo_sum - dense_lrmat_test) / normFrob(matrix_result_wo_sum);
    is_error = is_error || !(error < A_auto_approximation.get_epsilon() * (1 + additional_compression_tolerance));
    cout << "> Errors on a lrmat lrmat product to lrmat with auto approximation and without lrmat sum: " << error << endl;

    lrmat_test = C_auto_approximation;
    add_lrmat_lrmat_product(transa, transb, alpha, A_auto_approximation, B_auto_approximation, beta, lrmat_test);
    dense_lrmat_test.resize(lrmat_test.get_U().nb_rows(), lrmat_test.get_V().nb_cols());
    lrmat_test.copy_to_dense(dense_lrmat_test.data());
    error    = normFrob(matrix_result_w_lrmat_sum - dense_lrmat_test) / normFrob(matrix_result_w_lrmat_sum);
    is_error = is_error || !(error < A_auto_approximation.get_epsilon() * (1 + additional_compression_tolerance + additional_lrmat_sum_tolerance));
    cout << "> Errors on a lrmat lrmat product to lrmat with auto approximation and with lrmat sum: " << error << endl;

    cout << "test : " << is_error << endl
         << endl;

    return is_error;
}
