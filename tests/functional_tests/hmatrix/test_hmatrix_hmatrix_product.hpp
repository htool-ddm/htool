#include <htool/clustering/clustering.hpp>
#include <htool/hmatrix/hmatrix.hpp>
#include <htool/hmatrix/hmatrix_output.hpp>
#include <htool/hmatrix/linalg/interface.hpp>
#include <htool/hmatrix/lrmat/SVD.hpp>
#include <htool/hmatrix/lrmat/linalg/interface.hpp>
#include <htool/hmatrix/tree_builder/tree_builder.hpp>
#include <htool/matrix/linalg/interface.hpp>
#include <htool/testing/generate_test_case.hpp>
#include <htool/testing/generator_input.hpp>
#include <htool/testing/generator_test.hpp>
#include <htool/testing/geometry.hpp>
#include <htool/testing/partition.hpp>

using namespace std;
using namespace htool;

template <typename T, typename GeneratorTestType>
bool test_hmatrix_hmatrix_product(const TestCase<T, GeneratorTestType> &test_case, bool use_local_cluster, htool::underlying_type<T> epsilon, htool::underlying_type<T> margin) {
    // Logger::get_instance().set_current_log_level(LogLevel::INFO);

    int rankWorld;
    MPI_Comm_rank(MPI_COMM_WORLD, &rankWorld);
    bool is_error = false;
    double eta    = 10;
    char side     = test_case.side;
    char Symmetry = test_case.symmetry;
    char UPLO     = test_case.UPLO;
    char transa   = test_case.transa;

    char left_hmatrix_symmetry  = side == 'L' ? Symmetry : 'N';
    char left_hmatrix_UPLO      = side == 'L' ? UPLO : 'N';
    char right_hmatrix_symmetry = side == 'R' ? Symmetry : 'N';
    char right_hmatrix_UPLO     = side == 'R' ? UPLO : 'N';

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

    HMatrixTreeBuilder<T, htool::underlying_type<T>> hmatrix_tree_builder_A(*root_cluster_A_output, *root_cluster_A_input, epsilon, eta, left_hmatrix_symmetry, left_hmatrix_UPLO, -1, -1, rankWorld);
    HMatrixTreeBuilder<T, htool::underlying_type<T>> hmatrix_tree_builder_B(*root_cluster_B_output, *root_cluster_B_input, epsilon, eta, right_hmatrix_symmetry, right_hmatrix_UPLO, -1, -1, rankWorld);
    HMatrixTreeBuilder<T, htool::underlying_type<T>> hmatrix_tree_builder_C(*root_cluster_C_output, *root_cluster_C_input, epsilon, eta, 'N', 'N', -1, -1, rankWorld);
    hmatrix_tree_builder_C.set_minimal_source_depth(2);

    // build
    HMatrix<T, htool::underlying_type<T>> A = hmatrix_tree_builder_A.build(*test_case.operator_A);
    HMatrix<T, htool::underlying_type<T>> B = hmatrix_tree_builder_B.build(*test_case.operator_B);
    HMatrix<T, htool::underlying_type<T>> C = hmatrix_tree_builder_C.build(*test_case.operator_C);
    HMatrix<T, htool::underlying_type<T>> hmatrix_test(C);
    save_leaves_with_rank(A, "A_leaves_" + std::to_string(rankWorld));
    save_leaves_with_rank(B, "B_leaves_" + std::to_string(rankWorld));
    save_leaves_with_rank(C, "C_leaves_" + std::to_string(rankWorld));

    // Dense matrices
    int ni_A = root_cluster_A_input->get_size();
    int no_A = root_cluster_A_output->get_size();
    int ni_B = root_cluster_B_input->get_size();
    int no_B = root_cluster_B_output->get_size();
    int ni_C = root_cluster_C_input->get_size();
    int no_C = root_cluster_C_output->get_size();

    Matrix<T> A_dense(no_A, ni_A), B_dense(no_B, ni_B), C_dense(no_C, ni_C), matrix_test;
    test_case.operator_A->copy_submatrix(no_A, ni_A, root_cluster_A_output->get_offset(), root_cluster_A_input->get_offset(), A_dense.data());
    test_case.operator_B->copy_submatrix(no_B, ni_B, root_cluster_B_output->get_offset(), root_cluster_B_input->get_offset(), B_dense.data());
    test_case.operator_C->copy_submatrix(no_C, ni_C, root_cluster_C_output->get_offset(), root_cluster_C_input->get_offset(), C_dense.data());

    Matrix<T> HA_dense(A.get_target_cluster().get_size(), A.get_source_cluster().get_size());
    Matrix<T> HB_dense(B.get_target_cluster().get_size(), B.get_source_cluster().get_size());
    Matrix<T> HC_dense(C.get_target_cluster().get_size(), C.get_source_cluster().get_size());
    copy_to_dense(A, HA_dense.data());
    copy_to_dense(B, HB_dense.data());
    copy_to_dense(C, HC_dense.data());

    // lrmat
    SVD<T> compressor;
    htool::underlying_type<T> lrmat_tol = 1e-6;
    LowRankMatrix<T> C_auto_approximation(*test_case.operator_C, compressor, *root_cluster_C_output, *root_cluster_C_input, -1, lrmat_tol), lrmat_test(lrmat_tol);

    // Random Input
    T alpha(1), beta(0), scaling_coefficient;
    htool::underlying_type<T> error;
    generate_random_scalar(alpha);
    generate_random_scalar(beta);
    generate_random_scalar(scaling_coefficient);

    // References
    Matrix<T> matrix_result_w_matrix_sum(C_dense), densified_hmatrix_test(C_dense), matrix_result_w_lrmat_sum(C_dense), matrix_result_wo_sum(C_dense), dense_lrmat_test(C_dense), test(HC_dense);
    add_matrix_matrix_product(transa, 'N', alpha, A_dense, B_dense, beta, matrix_result_w_matrix_sum);
    add_matrix_matrix_product(transa, 'N', alpha, A_dense, B_dense, T(0), matrix_result_wo_sum);
    add_matrix_matrix_product(transa, 'N', alpha, HA_dense, HB_dense, beta, test);

    C_auto_approximation.copy_to_dense(matrix_result_w_lrmat_sum.data());
    add_matrix_matrix_product(transa, 'N', alpha, A_dense, B_dense, beta, matrix_result_w_lrmat_sum);

    // Product
    lrmat_test = C_auto_approximation;
    if (A.get_symmetry() == 'N') {
        add_hmatrix_hmatrix_product(transa, 'N', alpha, A, B, beta, lrmat_test);
    } else {
        add_hmatrix_hmatrix_product_symmetry('L', transa, 'N', alpha, A, B, beta, lrmat_test, A.get_UPLO(), A.get_symmetry());
    }
    dense_lrmat_test.resize(lrmat_test.get_U().nb_rows(), lrmat_test.get_V().nb_cols());
    lrmat_test.copy_to_dense(dense_lrmat_test.data());
    error    = normFrob(matrix_result_w_lrmat_sum - dense_lrmat_test) / normFrob(matrix_result_w_lrmat_sum);
    is_error = is_error || !(error < std::max(epsilon, lrmat_tol) * margin);
    cout << "> Errors on a hmatrix hmatrix product to lrmat: " << error << endl;

    matrix_test = C_dense;
    add_hmatrix_hmatrix_product(transa, 'N', alpha, A, B, beta, matrix_test);
    error    = normFrob(matrix_result_w_matrix_sum - matrix_test) / normFrob(matrix_result_w_matrix_sum);
    is_error = is_error || !(error < std::max(epsilon, lrmat_tol) * margin);
    cout << "> Errors on a hmatrix hmatrix product to matrix: " << error << endl;

    hmatrix_test = C;
    if (A.get_symmetry() == 'N') {
        add_hmatrix_hmatrix_product(transa, 'N', alpha, A, B, beta, hmatrix_test);
    } else {
        add_hmatrix_hmatrix_product_symmetry('L', transa, 'N', alpha, A, B, beta, hmatrix_test, A.get_UPLO(), A.get_symmetry());
    }
    copy_to_dense(hmatrix_test, densified_hmatrix_test.data());
    error    = normFrob(matrix_result_w_matrix_sum - densified_hmatrix_test) / normFrob(matrix_result_w_matrix_sum);
    is_error = is_error || !(error < epsilon * margin);
    cout << "> Errors on a hmatrix hmatrix product to hmatrix: " << error << endl;

    return is_error;
}
