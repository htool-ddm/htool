#include "test_lrmat_lrmat_product.hpp"         // for test_lr...
#include "test_lrmat_matrix_product.hpp"        // for test_lr...
#include "test_matrix_lrmat_product.hpp"        // for test_ma...
#include "test_matrix_matrix_product.hpp"       // for test_ma...
#include <array>                                // for array
#include <htool/hmatrix/lrmat/lrmat.hpp>        // for LowRank...
#include <htool/matrix/matrix.hpp>              // for Matrix
#include <htool/misc/misc.hpp>                  // for underly...
#include <htool/testing/generate_test_case.hpp> // for TestCas...
#include <htool/testing/generator_input.hpp>    // for generat...

using namespace std;
using namespace htool;

template <typename T, typename GeneratorTestType, class Compressor>
bool test_lrmat_product(char transa, char transb, int n1, int n2, int n3, htool::underlying_type<T> epsilon, htool::underlying_type<T> additional_compression_tolerance, std::array<htool::underlying_type<T>, 4> additional_lrmat_sum_tolerances) {
    bool is_error       = false;
    const int ndistance = 2;
    htool::underlying_type<T> distance[ndistance];
    distance[0] = 15;
    distance[1] = 40;
    for (int idist = 0; idist < ndistance; idist++) {
        TestCaseProduct<T, GeneratorTestType> test_case(transa, transb, n1, n2, n3, distance[idist], distance[idist] + 10);

        T alpha(1), beta(1), scaling_coefficient;
        generate_random_scalar(alpha);
        generate_random_scalar(beta);
        generate_random_scalar(scaling_coefficient);

        // lrmat
        Compressor compressor_A(*test_case.operator_A);
        LowRankMatrix<T> A_auto_approximation(test_case.root_cluster_A_output->get_size(), test_case.root_cluster_A_input->get_size(), epsilon);
        compressor_A.copy_low_rank_approximation(test_case.root_cluster_A_output->get_size(), test_case.root_cluster_A_input->get_size(), test_case.root_cluster_A_output->get_offset(), test_case.root_cluster_A_input->get_offset(), A_auto_approximation);
        Compressor compressor_B(*test_case.operator_B);
        LowRankMatrix<T> B_auto_approximation(test_case.root_cluster_B_output->get_size(), test_case.root_cluster_B_input->get_size(), epsilon);
        compressor_B.copy_low_rank_approximation(test_case.root_cluster_B_output->get_size(), test_case.root_cluster_B_input->get_size(), test_case.root_cluster_B_output->get_offset(), test_case.root_cluster_B_input->get_offset(), B_auto_approximation);
        Compressor compressor_C(*test_case.operator_C);
        LowRankMatrix<T> C_auto_approximation(test_case.root_cluster_C_output->get_size(), test_case.root_cluster_C_input->get_size(), epsilon);
        compressor_C.copy_low_rank_approximation(test_case.root_cluster_C_output->get_size(), test_case.root_cluster_C_input->get_size(), test_case.root_cluster_C_output->get_offset(), test_case.root_cluster_C_input->get_offset(), C_auto_approximation);

        // dense
        Matrix<T> A_dense(test_case.no_A, test_case.ni_A), B_dense(test_case.no_B, test_case.ni_B), C_dense(test_case.no_C, test_case.ni_C);
        test_case.operator_A->copy_submatrix(test_case.no_A, test_case.ni_A, 0, 0, A_dense.data());
        test_case.operator_B->copy_submatrix(test_case.no_B, test_case.ni_B, 0, 0, B_dense.data());
        test_case.operator_C->copy_submatrix(test_case.no_C, test_case.ni_C, 0, 0, C_dense.data());
        Matrix<T> matrix_result_w_matrix_sum(C_dense), matrix_result_wo_sum(C_dense), dense_lrmat_test, matrix_test, matrix_result_w_lrmat_sum(C_dense);
        C_auto_approximation.copy_to_dense(matrix_result_w_lrmat_sum.data());

        add_matrix_matrix_product(transa, transb, alpha, A_dense, B_dense, beta, matrix_result_w_matrix_sum);
        add_matrix_matrix_product(transa, transb, alpha, A_dense, B_dense, beta, matrix_result_w_lrmat_sum);
        add_matrix_matrix_product(transa, transb, alpha, A_dense, B_dense, T(0), matrix_result_wo_sum);

        is_error = is_error || test_lrmat_lrmat_product<T, GeneratorTestType, Compressor>(transa, transb, alpha, beta, A_auto_approximation, B_auto_approximation, C_auto_approximation, C_dense, matrix_result_w_matrix_sum, matrix_result_wo_sum, matrix_result_w_lrmat_sum, epsilon, additional_compression_tolerance, additional_lrmat_sum_tolerances[0]);
        is_error = is_error || test_lrmat_matrix_product<T, GeneratorTestType, Compressor>(transa, transb, alpha, beta, scaling_coefficient, A_auto_approximation, C_auto_approximation, B_dense, C_dense, matrix_result_w_matrix_sum, matrix_result_wo_sum, matrix_result_w_lrmat_sum, epsilon, additional_compression_tolerance, additional_lrmat_sum_tolerances[1]);
        is_error = is_error || test_matrix_lrmat_product<T, GeneratorTestType, Compressor>(transa, transb, alpha, beta, B_auto_approximation, C_auto_approximation, A_dense, C_dense, matrix_result_w_matrix_sum, matrix_result_wo_sum, matrix_result_w_lrmat_sum, epsilon, additional_compression_tolerance, additional_lrmat_sum_tolerances[2]);
        is_error = is_error || test_matrix_matrix_product<T, GeneratorTestType, Compressor>(transa, transb, alpha, beta, C_auto_approximation, A_dense, B_dense, matrix_result_wo_sum, matrix_result_w_lrmat_sum, epsilon, additional_compression_tolerance, additional_lrmat_sum_tolerances[3]);
    }

    return is_error;
}
