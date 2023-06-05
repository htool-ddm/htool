#include <htool/hmatrix/hmatrix.hpp>                           // for HMatrix
#include <htool/hmatrix/linalg/add_matrix_hmatrix_product.hpp> // for add_m...
#include <htool/hmatrix/lrmat/SVD.hpp>                         // for SVD
#include <htool/hmatrix/lrmat/lrmat.hpp>                       // for LowRa...
#include <htool/hmatrix/tree_builder/tree_builder.hpp>         // for HMatr...
#include <htool/matrix/matrix.hpp>                             // for Matrix
#include <htool/misc/misc.hpp>                                 // for under...
#include <htool/testing/generator_input.hpp>                   // for gener...
#include <iostream>                                            // for opera...
#include <memory>                                              // for make_...
#include <mpi.h>                                               // for MPI_C...
#include <vector>                                              // for vector
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
bool test_matrix_hmatrix_product(const TestCaseProduct<T, GeneratorTestType> &test_case, bool use_local_cluster, htool::underlying_type<T> epsilon, htool::underlying_type<T> additional_lrmat_sum_tolerance) {

    int rankWorld;
    MPI_Comm_rank(MPI_COMM_WORLD, &rankWorld);
    bool is_error = false;
    double eta    = 1;
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

    HMatrixTreeBuilder<T, htool::underlying_type<T>> hmatrix_tree_builder(*root_cluster_B_output, *root_cluster_B_input, epsilon, eta, 'N', 'N', -1, -1, rankWorld);
    hmatrix_tree_builder.set_low_rank_generator(std::make_shared<SVD<T>>());

    // build
    HMatrix<T, htool::underlying_type<T>> root_hmatrix = hmatrix_tree_builder.build(*test_case.operator_B);

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

    Matrix<T> HB_dense(root_hmatrix.get_target_cluster().get_size(), root_hmatrix.get_source_cluster().get_size());
    copy_to_dense(root_hmatrix, HB_dense.data());

    // lrmat
    SVD<T> compressor;
    htool::underlying_type<T> lrmat_tolerance = 0.0001;
    LowRankMatrix<T> C_auto_approximation(*test_case.operator_C, compressor, *root_cluster_C_output, *root_cluster_C_input, -1, lrmat_tolerance), lrmat_test(lrmat_tolerance);

    // Random Input matrix
    std::vector<T> B_vec, C_vec, test_vec;
    B_vec = B_dense.get_col(0);
    C_vec = C_dense.get_col(0);
    T alpha(1), beta(1), scaling_coefficient;
    htool::underlying_type<T> error;
    generate_random_scalar(alpha);
    generate_random_scalar(beta);
    generate_random_scalar(scaling_coefficient);

    // References
    Matrix<T> matrix_result_w_matrix_sum(C_dense), matrix_result_wo_sum(C_dense), matrix_result_w_lrmat_sum(C_dense);
    add_matrix_matrix_product(transa, transb, alpha, A_dense, B_dense, beta, matrix_result_w_matrix_sum);
    add_matrix_matrix_product(transa, transb, alpha, A_dense, B_dense, T(0), matrix_result_wo_sum);
    add_matrix_matrix_product(transa, transb, alpha, A_dense, HB_dense, beta, matrix_result_w_lrmat_sum);

    // Products
    matrix_test = C_dense;
    internal_add_matrix_hmatrix_product(transa, transb, alpha, A_dense, root_hmatrix, beta, matrix_test);
    error    = normFrob(matrix_result_w_matrix_sum - matrix_test) / normFrob(matrix_result_w_matrix_sum);
    is_error = is_error || !(error < epsilon);
    cout << "> Errors on a matrix hmatrix product: " << error << endl;

    lrmat_test = C_auto_approximation;
    internal_add_matrix_hmatrix_product(transa, transb, alpha, A_dense, root_hmatrix, T(0), lrmat_test);
    dense_lrmat_test.resize(lrmat_test.get_U().nb_rows(), lrmat_test.get_V().nb_cols());
    lrmat_test.copy_to_dense(dense_lrmat_test.data());
    error    = normFrob(matrix_result_wo_sum - dense_lrmat_test) / normFrob(matrix_result_wo_sum);
    is_error = is_error || !(error < std::max(epsilon, lrmat_tolerance));
    cout << "> Errors on a matrix hmatrix product to lrmat without sum: " << error << endl;

    lrmat_test = C_auto_approximation;
    internal_add_matrix_hmatrix_product(transa, transb, alpha, A_dense, root_hmatrix, beta, lrmat_test);
    dense_lrmat_test.resize(lrmat_test.get_U().nb_rows(), lrmat_test.get_V().nb_cols());
    lrmat_test.copy_to_dense(dense_lrmat_test.data());
    error    = normFrob(matrix_result_w_lrmat_sum - dense_lrmat_test) / normFrob(matrix_result_w_lrmat_sum);
    is_error = is_error || !(error < std::max(epsilon, lrmat_tolerance) * (1 + additional_lrmat_sum_tolerance));
    cout << "> Errors on a matrix hmatrix product to lrmat with sum: " << error << endl;

    return is_error;
}
