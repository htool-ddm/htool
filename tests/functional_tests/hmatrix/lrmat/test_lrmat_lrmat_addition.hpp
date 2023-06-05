#include <htool/hmatrix/lrmat/linalg/add_lrmat_lrmat.hpp> // for add_lrmat_...
#include <htool/hmatrix/lrmat/lrmat.hpp>                  // for LowRankMatrix
#include <htool/matrix/matrix.hpp>                        // for normFrob
#include <htool/misc/misc.hpp>                            // for underlying...
#include <htool/testing/generate_test_case.hpp>           // for TestCaseAd...
#include <iostream>                                       // for operator<<
#include <memory>                                         // for make_unique

using namespace std;
using namespace htool;

template <typename T, typename GeneratorTestType, class Compressor>
bool test_lrmat_lrmat_addition(int n1, int n2, htool::underlying_type<T> epsilon, htool::underlying_type<T> margin) {
    bool is_error = false;
    TestCaseAddition<T, GeneratorTestType> test_case(n1, n2, 2);

    // lrmat
    Compressor compressor;
    LowRankMatrix<T> A_approximation(*test_case.operator_A, compressor, *test_case.root_cluster_A_output, *test_case.root_cluster_A_input, -1, epsilon);
    LowRankMatrix<T> zero_A_approximation(*test_case.operator_A, compressor, *test_case.root_cluster_A_output, *test_case.root_cluster_A_input, 0, epsilon);
    LowRankMatrix<T> sub_zero_A_approximation(*test_case.operator_A, compressor, *test_case.root_cluster_B_output, *test_case.root_cluster_B_input, 0, epsilon);

    std::unique_ptr<LowRankMatrix<T>> sub_A_approximation_ptr = std::make_unique<LowRankMatrix<T>>(*test_case.operator_A, compressor, *test_case.root_cluster_B_output, *test_case.root_cluster_B_input, -1, epsilon);
    if (sub_A_approximation_ptr->rank_of() == 0) {
        sub_A_approximation_ptr = std::make_unique<LowRankMatrix<T>>(*test_case.operator_A, compressor, *test_case.root_cluster_B_output, *test_case.root_cluster_B_input, (test_case.root_cluster_B_output->get_size() * test_case.root_cluster_B_input->get_size()) / (test_case.root_cluster_B_output->get_size() + test_case.root_cluster_B_input->get_size()), epsilon);
    }
    LowRankMatrix<T> &sub_A_approximation = *sub_A_approximation_ptr;

    // Reference
    Matrix<T> A_dense(A_approximation.nb_rows(), A_approximation.nb_cols(), 0);
    A_approximation.copy_to_dense(A_dense.data());
    Matrix<T> sub_A_dense(sub_A_approximation.nb_rows(), sub_A_approximation.nb_cols(), 0);
    sub_A_approximation.copy_to_dense(sub_A_dense.data());
    Matrix<T> sub_A_dense_extended(A_approximation.nb_rows(), A_approximation.nb_cols(), 0);

    for (int i = 0; i < sub_A_dense.nb_rows(); i++) {
        for (int j = 0; j < sub_A_dense.nb_cols(); j++) {
            sub_A_dense_extended(i + test_case.root_cluster_B_output->get_offset(), j + test_case.root_cluster_B_input->get_offset()) = sub_A_dense(i, j);
        }
    }
    Matrix<T> matrix_result_w_sum, matrix_result_wo_sum, matrix_test(A_dense), sub_matrix_test(sub_A_dense);
    matrix_result_w_sum  = A_dense + sub_A_dense_extended;
    matrix_result_wo_sum = sub_A_dense_extended;

    // Addition
    htool::underlying_type<T> error;
    LowRankMatrix<T> lrmat_test(epsilon);
    lrmat_test = A_approximation;
    add_lrmat_lrmat(sub_A_approximation, *test_case.root_cluster_B_output, *test_case.root_cluster_B_input, lrmat_test, *test_case.root_cluster_A_output, *test_case.root_cluster_A_input);
    lrmat_test.copy_to_dense(matrix_test.data());
    error    = normFrob(matrix_result_w_sum - matrix_test) / normFrob(matrix_result_w_sum);
    is_error = is_error || !(error < epsilon);
    cout << "> Errors on a smaller lrmat addition to another lrmat and with sum: " << error << endl;

    lrmat_test = zero_A_approximation;
    add_lrmat_lrmat(sub_A_approximation, *test_case.root_cluster_B_output, *test_case.root_cluster_B_input, lrmat_test, *test_case.root_cluster_A_output, *test_case.root_cluster_A_input);
    lrmat_test.copy_to_dense(matrix_test.data());
    error    = normFrob(matrix_result_wo_sum - matrix_test) / normFrob(matrix_result_wo_sum);
    is_error = is_error || !(error < epsilon);
    cout << "> Errors on a smaller lrmat addition to another lrmat and without sum: " << error << endl;

    lrmat_test = sub_A_approximation;
    add_lrmat_lrmat(A_approximation, *test_case.root_cluster_A_output, *test_case.root_cluster_A_input, lrmat_test, *test_case.root_cluster_B_output, *test_case.root_cluster_B_input);
    lrmat_test.copy_to_dense(sub_matrix_test.data());
    error    = normFrob(2 * sub_A_dense - sub_matrix_test) / normFrob(2 * sub_A_dense);
    is_error = is_error || !(error < epsilon * margin);
    cout << "> Errors on a larger lrmat addition with another lrmat and with sum: " << error << endl;

    lrmat_test = sub_zero_A_approximation;
    add_lrmat_lrmat(A_approximation, *test_case.root_cluster_A_output, *test_case.root_cluster_A_input, lrmat_test, *test_case.root_cluster_B_output, *test_case.root_cluster_B_input);
    lrmat_test.copy_to_dense(sub_matrix_test.data());
    error    = normFrob(sub_A_dense - sub_matrix_test) / normFrob(sub_A_dense);
    is_error = is_error || !(error < epsilon * margin);
    cout << "> Errors on a larger lrmat addition to another lrmat and without sum: " << error << endl;

    return is_error;
}
