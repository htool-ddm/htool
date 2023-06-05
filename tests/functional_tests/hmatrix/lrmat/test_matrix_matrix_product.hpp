#include <htool/hmatrix/lrmat/linalg/add_matrix_matrix_product.hpp>
#include <htool/hmatrix/lrmat/lrmat.hpp> // for LowRank...
#include <htool/matrix/matrix.hpp>       // for normFrob
#include <htool/misc/misc.hpp>           // for underly...
#include <iostream>                      // for basic_o...

using namespace std;
using namespace htool;

template <typename T, typename GeneratorTestType, class Compressor>
bool test_matrix_matrix_product(char transa, char transb, T alpha, T beta, const LowRankMatrix<T> &C_auto_approximation, const Matrix<T> &A_dense, const Matrix<T> &B_dense, const Matrix<T> &matrix_result_wo_sum, const Matrix<T> &matrix_result_w_lrmat_sum, htool::underlying_type<T> epsilon, htool::underlying_type<T> additional_compression_tolerance, htool::underlying_type<T> additional_lrmat_sum_tolerance) {
    bool is_error = false;
    LowRankMatrix<T> lrmat_test(epsilon);
    Matrix<T> matrix_test, dense_lrmat_test;
    htool::underlying_type<T> error;

    // Product
    lrmat_test = C_auto_approximation;
    add_matrix_matrix_product(transa, transb, alpha, A_dense, B_dense, T(0), lrmat_test);
    dense_lrmat_test.resize(lrmat_test.get_U().nb_rows(), lrmat_test.get_V().nb_cols());
    lrmat_test.copy_to_dense(dense_lrmat_test.data());
    error    = normFrob(matrix_result_wo_sum - dense_lrmat_test) / normFrob(matrix_result_wo_sum);
    is_error = is_error || !(error < epsilon);
    cout << "> Errors on a matrix matrix product to lrmat without sum: " << error << " vs " << epsilon << endl;

    lrmat_test = C_auto_approximation;
    add_matrix_matrix_product(transa, transb, alpha, A_dense, B_dense, beta, lrmat_test);
    dense_lrmat_test.resize(lrmat_test.get_U().nb_rows(), lrmat_test.get_V().nb_cols());
    lrmat_test.copy_to_dense(dense_lrmat_test.data());
    error    = normFrob(matrix_result_w_lrmat_sum - dense_lrmat_test) / normFrob(matrix_result_w_lrmat_sum);
    is_error = is_error || !(error < epsilon * (1 + additional_compression_tolerance + additional_lrmat_sum_tolerance));
    cout << "> Errors on a matrix matrix product to lrmat with sum: " << error << " vs " << epsilon * (1 + additional_compression_tolerance + additional_lrmat_sum_tolerance) << endl;

    cout << "test : " << is_error << endl
         << endl;

    return is_error;
}
