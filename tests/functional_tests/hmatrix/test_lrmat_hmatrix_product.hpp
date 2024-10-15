#include <htool/hmatrix/hmatrix.hpp>
#include <htool/hmatrix/linalg/add_lrmat_hmatrix_product.hpp>
#include <htool/hmatrix/lrmat/SVD.hpp> // for SVD
#include <htool/hmatrix/lrmat/linalg/add_lrmat_matrix_product.hpp>
#include <htool/hmatrix/lrmat/lrmat.hpp>
#include <htool/hmatrix/tree_builder/tree_builder.hpp>
#include <htool/matrix/matrix.hpp>
#include <htool/misc/misc.hpp>
#include <htool/testing/generator_input.hpp>
#include <iostream>
#include <memory>
#include <mpi.h>
namespace htool {
template <typename CoordinatesPrecision>
class Cluster;
}
namespace htool {
template <typename T, typename GeneratorTestType>
class TestCaseProduct;
}
using namespace std;
using namespace htool;

template <typename T, typename GeneratorTestType>
bool test_lrmat_hmatrix_product(const TestCaseProduct<T, GeneratorTestType> &test_case, bool use_local_cluster, htool::underlying_type<T> epsilon, htool::underlying_type<T> additional_lrmat_sum_tolerance) {

    int rankWorld;
    MPI_Comm_rank(MPI_COMM_WORLD, &rankWorld);
    bool is_error = false;
    double eta    = 10;
    char transa   = test_case.transa;
    char transb   = test_case.transb;

    const Cluster<htool::underlying_type<T>> *root_cluster_A_output, *root_cluster_A_input, *root_cluster_B_output, *root_cluster_B_input, *root_cluster_C_output, *root_cluster_C_input;
    if (use_local_cluster) {
        root_cluster_A_output = &test_case.root_cluster_A_output->get_cluster_on_partition(rankWorld);
        root_cluster_A_input  = &test_case.root_cluster_A_input->get_cluster_on_partition(rankWorld);
        root_cluster_B_output = &test_case.root_cluster_B_output->get_cluster_on_partition(rankWorld);
        root_cluster_B_input  = &test_case.root_cluster_B_input->get_cluster_on_partition(rankWorld);
        root_cluster_C_output = &test_case.root_cluster_C_output->get_cluster_on_partition(rankWorld);
        root_cluster_C_input  = &test_case.root_cluster_C_input->get_cluster_on_partition(rankWorld);
    } else {
        if (transa == 'N') {
            root_cluster_A_output = &test_case.root_cluster_A_output->get_cluster_on_partition(rankWorld);
            root_cluster_A_input  = test_case.root_cluster_A_input;
            root_cluster_B_output = test_case.root_cluster_B_output;
            root_cluster_B_input  = test_case.root_cluster_B_input;
            root_cluster_C_output = &test_case.root_cluster_C_output->get_cluster_on_partition(rankWorld);
            root_cluster_C_input  = test_case.root_cluster_C_input;
        } else {
            root_cluster_A_output = test_case.root_cluster_A_output;
            root_cluster_A_input  = &test_case.root_cluster_A_input->get_cluster_on_partition(rankWorld);
            root_cluster_B_output = test_case.root_cluster_B_output;
            root_cluster_B_input  = test_case.root_cluster_B_input;
            root_cluster_C_output = &test_case.root_cluster_C_output->get_cluster_on_partition(rankWorld);
            root_cluster_C_input  = test_case.root_cluster_C_input;
        }
    }

    HMatrixTreeBuilder<T, htool::underlying_type<T>> hmatrix_tree_builder(epsilon, eta, 'N', 'N', -1, std::make_shared<SVD<T>>());

    HMatrixTreeBuilder<T, htool::underlying_type<T>> hmatrix_tree_builder_C(epsilon, eta, 'N', 'N', -1, std::make_shared<SVD<T>>());

    // build
    HMatrix<T, htool::underlying_type<T>> root_hmatrix = hmatrix_tree_builder.build(*test_case.operator_B, *root_cluster_B_output, *root_cluster_B_input);
    HMatrix<T, htool::underlying_type<T>> C            = hmatrix_tree_builder_C.build(*test_case.operator_C, *root_cluster_C_output, *root_cluster_C_input);
    HMatrix<T, htool::underlying_type<T>> hmatrix_test(C);

    // Dense matrix
    int ni_A = root_cluster_A_input->get_size();
    int no_A = root_cluster_A_output->get_size();
    int ni_B = root_cluster_B_input->get_size();
    int no_B = root_cluster_B_output->get_size();
    int ni_C = root_cluster_C_input->get_size();
    int no_C = root_cluster_C_output->get_size();

    Matrix<T> A_dense(no_A, ni_A), B_dense(no_B, ni_B), C_dense(no_C, ni_C);
    test_case.operator_A->copy_submatrix(no_A, ni_A, root_cluster_A_output->get_offset(), root_cluster_A_input->get_offset(), A_dense.data());
    test_case.operator_B->copy_submatrix(no_B, ni_B, root_cluster_B_output->get_offset(), root_cluster_B_input->get_offset(), B_dense.data());
    test_case.operator_C->copy_submatrix(no_C, ni_C, root_cluster_C_output->get_offset(), root_cluster_C_input->get_offset(), C_dense.data());
    Matrix<T> matrix_test, dense_lrmat_test;

    // lrmat
    SVD<T> compressor;
    htool::underlying_type<T> lrmat_tolerance = 1e-6;
    // std::unique_ptr<LowRankMatrix<T>> A_auto_approximation, C_auto_approximation;
    LowRankMatrix<T> lrmat_test(epsilon);
    LowRankMatrix<T> A_auto_approximation(*test_case.operator_A, compressor, *root_cluster_A_output, *root_cluster_A_input, -1, lrmat_tolerance);
    LowRankMatrix<T> C_auto_approximation(*test_case.operator_C, compressor, *root_cluster_C_output, *root_cluster_C_input, -1, lrmat_tolerance);

    // Random Input matrix
    T alpha(1), beta(1), scaling_coefficient;
    htool::underlying_type<T> error;
    generate_random_scalar(alpha);
    generate_random_scalar(beta);
    generate_random_scalar(scaling_coefficient);

    // References
    Matrix<T> matrix_result_w_matrix_sum(C_dense), matrix_result_wo_sum(C_dense), matrix_result_w_lrmat_sum(C_dense), densified_hmatrix_test(C_dense);
    C_auto_approximation.copy_to_dense(matrix_result_w_lrmat_sum.data());
    add_lrmat_matrix_product(transa, transb, alpha, A_auto_approximation, B_dense, beta, matrix_result_w_matrix_sum);
    add_lrmat_matrix_product(transa, transb, alpha, A_auto_approximation, B_dense, beta, matrix_result_w_lrmat_sum);
    add_lrmat_matrix_product(transa, transb, alpha, A_auto_approximation, B_dense, T(0), matrix_result_wo_sum);

    // Product
    matrix_test = C_dense;
    internal_add_lrmat_hmatrix_product(transa, transb, alpha, A_auto_approximation, root_hmatrix, beta, matrix_test);
    error    = normFrob(matrix_result_w_matrix_sum - matrix_test) / normFrob(matrix_result_w_matrix_sum);
    is_error = is_error || !(error < epsilon);
    cout << "> Errors on a lrmat hmatrix product to matrix: " << error << endl;

    lrmat_test = C_auto_approximation;
    internal_add_lrmat_hmatrix_product(transa, transb, alpha, A_auto_approximation, root_hmatrix, T(0), lrmat_test);
    dense_lrmat_test.resize(lrmat_test.get_U().nb_rows(), lrmat_test.get_V().nb_cols());
    lrmat_test.copy_to_dense(dense_lrmat_test.data());
    error    = normFrob(matrix_result_wo_sum - dense_lrmat_test) / normFrob(matrix_result_wo_sum);
    is_error = is_error || !(error < epsilon);
    cout << "> Errors on a lrmat hmatrix product to lrmat without lrmat sum: " << error << endl;

    lrmat_test = C_auto_approximation;
    internal_add_lrmat_hmatrix_product(transa, transb, alpha, A_auto_approximation, root_hmatrix, beta, lrmat_test);
    dense_lrmat_test.resize(lrmat_test.get_U().nb_rows(), lrmat_test.get_V().nb_cols());
    lrmat_test.copy_to_dense(dense_lrmat_test.data());
    error    = normFrob(matrix_result_w_lrmat_sum - dense_lrmat_test) / normFrob(matrix_result_w_lrmat_sum);
    is_error = is_error || !(error < std::max(epsilon, lrmat_tolerance) * (1 + additional_lrmat_sum_tolerance));
    cout << "> Errors on a lrmat hmatrix product to lrmat with lrmat sum: " << error << endl;

    hmatrix_test = C;
    internal_add_lrmat_hmatrix_product(transa, transb, alpha, A_auto_approximation, root_hmatrix, T(0), hmatrix_test);
    copy_to_dense(hmatrix_test, densified_hmatrix_test.data());
    error    = normFrob(matrix_result_wo_sum - densified_hmatrix_test) / normFrob(matrix_result_wo_sum);
    is_error = is_error || !(error < epsilon);
    cout << "> Errors on a lrmat hmatrix product to hmatrix without lrmat sum: " << error << endl;

    hmatrix_test = C;
    internal_add_lrmat_hmatrix_product(transa, transb, alpha, A_auto_approximation, root_hmatrix, beta, hmatrix_test);
    copy_to_dense(hmatrix_test, densified_hmatrix_test.data());
    error    = normFrob(matrix_result_w_matrix_sum - densified_hmatrix_test) / normFrob(matrix_result_w_matrix_sum);
    is_error = is_error || !(error < std::max(epsilon, lrmat_tolerance) * (1 + additional_lrmat_sum_tolerance));
    cout << "> Errors on a lrmat hmatrix product to hmatrix with lrmat sum: " << error << endl;

    cout << "> is_error: " << is_error << endl;
    return is_error;
}
