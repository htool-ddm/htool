#include <htool/hmatrix/lrmat/linalg/add_lrmat_lrmat.hpp>
#include <htool/hmatrix/lrmat/lrmat.hpp>
#include <htool/testing/generate_test_case.hpp>
using namespace std;
using namespace htool;

template <typename T, typename GeneratorTestType, class Compressor>
bool test_lrmat_lrmat_addition(char transa, char transb, int n1, int n2, int n3, htool::underlying_type<T> epsilon) {
    bool is_error = false;
    TestCase<T, GeneratorTestType> test_case(transa, transb, n1, n2, n3, 2, 4, 'N', 'N', 'N');

    // lrmat
    Compressor compressor;
    LowRankMatrix<T> A_approximation(*test_case.operator_A, compressor, *test_case.root_cluster_A_output, *test_case.root_cluster_A_input, -1, epsilon);
    LowRankMatrix<T> zero_A_approximation(*test_case.operator_A, compressor, *test_case.root_cluster_A_output, *test_case.root_cluster_A_input, 0, epsilon);

    // Sub lrmat two level deep
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(0, 1);
    int first_target_index  = dist(gen);
    int second_target_index = dist(gen);
    int first_source_index  = dist(gen);
    int second_source_index = dist(gen);

    const Cluster<htool::underlying_type<T>> &sub_target_cluster = *test_case.root_cluster_A_output->get_children()[first_target_index]->get_children()[second_target_index];
    const Cluster<htool::underlying_type<T>> &sub_source_cluster = *test_case.root_cluster_A_input->get_children()[first_source_index]->get_children()[second_source_index];

    LowRankMatrix<T> sub_A_approximation(*test_case.operator_A, compressor, sub_target_cluster, sub_source_cluster, -1, epsilon);

    // Reference
    Matrix<T> A_dense(A_approximation.nb_rows(), A_approximation.nb_cols(), 0);
    A_approximation.copy_to_dense(A_dense.data());
    Matrix<T> sub_A_dense(sub_A_approximation.nb_rows(), sub_A_approximation.nb_cols(), 0);
    sub_A_approximation.copy_to_dense(sub_A_dense.data());
    Matrix<T> sub_A_dense_extended(A_approximation.nb_rows(), A_approximation.nb_cols(), 0);

    for (int i = 0; i < sub_A_dense.nb_rows(); i++) {
        for (int j = 0; j < sub_A_dense.nb_cols(); j++) {
            sub_A_dense_extended(i + sub_target_cluster.get_offset(), j + sub_source_cluster.get_offset()) = sub_A_dense(i, j);
        }
    }
    Matrix<T> matrix_result_w_sum, matrix_result_wo_sum, matrix_test(A_dense);
    matrix_result_w_sum  = A_dense + sub_A_dense_extended;
    matrix_result_wo_sum = sub_A_dense_extended;

    // Addition
    htool::underlying_type<T> error;
    add_lrmat_lrmat(A_approximation, *test_case.root_cluster_A_output, *test_case.root_cluster_A_input, sub_A_approximation, sub_target_cluster, sub_source_cluster);
    A_approximation.copy_to_dense(matrix_test.data());
    error    = normFrob(matrix_result_w_sum - matrix_test) / normFrob(matrix_result_w_sum);
    is_error = is_error || !(error < epsilon);
    cout << "> Errors on a lrmat lrmat addition with sum: " << error << endl;

    add_lrmat_lrmat(zero_A_approximation, *test_case.root_cluster_A_output, *test_case.root_cluster_A_input, sub_A_approximation, sub_target_cluster, sub_source_cluster);
    zero_A_approximation.copy_to_dense(matrix_test.data());
    error    = normFrob(matrix_result_wo_sum - matrix_test) / normFrob(matrix_result_wo_sum);
    is_error = is_error || !(error < epsilon);
    cout << "> Errors on a lrmat lrmat addition without sum: " << error << endl;

    return is_error;
}
