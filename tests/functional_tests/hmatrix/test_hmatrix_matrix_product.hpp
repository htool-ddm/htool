#include <htool/basic_types/vector.hpp>
#include <htool/hmatrix/hmatrix.hpp>
#include <htool/hmatrix/linalg/add_hmatrix_matrix_product.hpp>
#include <htool/hmatrix/linalg/add_hmatrix_matrix_product_row_major.hpp>
#include <htool/hmatrix/linalg/add_hmatrix_vector_product.hpp>
#include <htool/hmatrix/linalg/add_matrix_hmatrix_product.hpp>
#include <htool/hmatrix/lrmat/SVD.hpp>
#include <htool/hmatrix/lrmat/lrmat.hpp>
#include <htool/hmatrix/tree_builder/tree_builder.hpp>
#include <htool/matrix/linalg/add_matrix_matrix_product.hpp>
#include <htool/matrix/linalg/transpose.hpp>
#include <htool/matrix/matrix.hpp>
#include <htool/misc/misc.hpp>
#include <htool/testing/generator_input.hpp>
#include <iostream>
#include <memory>
#include <mpi.h>
#include <vector>
namespace htool {
template <typename CoordinatesPrecision>
class Cluster;
}
namespace htool {
template <typename T, typename GeneratorTestType>
class TestCaseProduct;
}
namespace htool {
template <typename T, typename GeneratorTestType>
class TestCaseSymmetricProduct;
}

using namespace std;
using namespace htool;

template <typename T, typename GeneratorTestType>
bool test_hmatrix_matrix_product(const TestCaseProduct<T, GeneratorTestType> &test_case, bool use_local_cluster, htool::underlying_type<T> epsilon, htool::underlying_type<T> additional_lrmat_sum_tolerance) {

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

    HMatrixTreeBuilder<T, htool::underlying_type<T>> hmatrix_tree_builder(epsilon, eta, 'N', 'N');
    hmatrix_tree_builder.set_low_rank_generator(std::make_shared<SVD<T>>(*test_case.operator_A));

    // build
    HMatrix<T, htool::underlying_type<T>> root_hmatrix = hmatrix_tree_builder.build(*test_case.operator_A, *root_cluster_A_output, *root_cluster_A_input);

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
    Matrix<T> matrix_test, dense_lrmat_test, transposed_B_dense(B_dense.nb_cols(), B_dense.nb_rows()), transposed_C_dense(C_dense.nb_cols(), C_dense.nb_rows());
    transpose(B_dense, transposed_B_dense);
    transpose(C_dense, transposed_C_dense);

    Matrix<T> HA_dense(root_hmatrix.get_target_cluster().get_size(), root_hmatrix.get_source_cluster().get_size());
    copy_to_dense(root_hmatrix, HA_dense.data());

    // lrmat
    SVD<T> compressor(*test_case.operator_C);
    htool::underlying_type<T> lrmat_tolerance = 0.0001;
    LowRankMatrix<T> C_auto_approximation(root_cluster_C_output->get_size(), root_cluster_C_input->get_size(), lrmat_tolerance), lrmat_test(root_cluster_C_output->get_size(), root_cluster_C_input->get_size(), lrmat_tolerance);
    compressor.copy_low_rank_approximation(root_cluster_C_output->get_size(), root_cluster_C_input->get_size(), root_cluster_C_output->get_offset(), root_cluster_C_input->get_offset(), C_auto_approximation);

    // Random Input matrix
    std::vector<T> B_vec, C_vec, test_vec;
    B_vec = get_col(B_dense, 0);
    C_vec = get_col(C_dense, 0);
    T alpha(0), beta(0), scaling_coefficient;
    htool::underlying_type<T> error;
    generate_random_scalar(alpha);
    generate_random_scalar(beta);
    generate_random_scalar(scaling_coefficient);

    // References
    Matrix<T> matrix_result_w_matrix_sum(C_dense), transposed_matrix_result_w_matrix_sum(C_dense.nb_cols(), C_dense.nb_rows()), matrix_result_wo_sum(C_dense), matrix_result_w_lrmat_sum(C_dense);
    C_auto_approximation.copy_to_dense(matrix_result_w_lrmat_sum.data());

    add_matrix_matrix_product(transa, transb, alpha, A_dense, B_dense, beta, matrix_result_w_matrix_sum);
    transpose(matrix_result_w_matrix_sum, transposed_matrix_result_w_matrix_sum);
    add_matrix_matrix_product(transa, transb, alpha, A_dense, B_dense, T(0), matrix_result_wo_sum);
    add_matrix_matrix_product(transa, transb, alpha, HA_dense, B_dense, beta, matrix_result_w_lrmat_sum);

    Matrix<T> scaled_matrix_result_w_matrix_sum(matrix_result_w_matrix_sum);
    scale(scaling_coefficient, scaled_matrix_result_w_matrix_sum);

    // Product
    if (transb == 'N') {
        test_vec = C_vec;
        internal_add_hmatrix_vector_product(transa, alpha, root_hmatrix, B_vec.data(), beta, test_vec.data());
        error    = norm2(get_col(matrix_result_w_matrix_sum, 0) - test_vec) / norm2(get_col(matrix_result_w_matrix_sum, 0));
        is_error = is_error || !(error < epsilon);
        cout << "> Errors on a hmatrix vector product: " << error << endl;

        test_vec = C_vec;
        openmp_internal_add_hmatrix_vector_product(transa, alpha, root_hmatrix, B_vec.data(), beta, test_vec.data());
        error    = norm2(get_col(matrix_result_w_matrix_sum, 0) - test_vec) / norm2(get_col(matrix_result_w_matrix_sum, 0));
        is_error = is_error || !(error < epsilon);
        cout << "> Errors on a openmp hmatrix vector product: " << error << endl;
    }

    matrix_test = C_dense;
    internal_add_hmatrix_matrix_product(transa, transb, alpha, root_hmatrix, B_dense, beta, matrix_test);
    error    = normFrob(matrix_result_w_matrix_sum - matrix_test) / normFrob(matrix_result_w_matrix_sum);
    is_error = is_error || !(error < epsilon);
    cout << "> Errors on a hmatrix matrix product: " << error << endl;

    if (transb == 'N') {
        matrix_test = transposed_C_dense;
        openmp_internal_add_hmatrix_matrix_product_row_major(transa, transb, alpha, root_hmatrix, transposed_B_dense.data(), beta, matrix_test.data(), C_dense.nb_cols());
        error    = normFrob(transposed_matrix_result_w_matrix_sum - matrix_test) / normFrob(transposed_matrix_result_w_matrix_sum);
        is_error = is_error || !(error < epsilon);
        cout << "> Errors on a openmp hmatrix matrix product with row major input: " << error << endl;

        matrix_test = transposed_C_dense;
        sequential_internal_add_hmatrix_matrix_product_row_major(transa, transb, alpha, root_hmatrix, transposed_B_dense.data(), beta, matrix_test.data(), C_dense.nb_cols());
        error    = normFrob(transposed_matrix_result_w_matrix_sum - matrix_test) / normFrob(transposed_matrix_result_w_matrix_sum);
        is_error = is_error || !(error < epsilon);
        cout << "> Errors on a sequential hmatrix matrix product with row major input: " << error << endl;
    }

    // Product
    lrmat_test = C_auto_approximation;
    internal_add_hmatrix_matrix_product(transa, transb, alpha, root_hmatrix, B_dense, T(0), lrmat_test);
    dense_lrmat_test.resize(lrmat_test.get_U().nb_rows(), lrmat_test.get_V().nb_cols());
    lrmat_test.copy_to_dense(dense_lrmat_test.data());
    error    = normFrob(matrix_result_wo_sum - dense_lrmat_test) / normFrob(matrix_result_wo_sum);
    is_error = is_error || !(error < std::max(epsilon, lrmat_tolerance));
    cout << "> Errors on a hmatrix matrix product to lrmat without sum: " << error << endl;

    lrmat_test = C_auto_approximation;
    internal_add_hmatrix_matrix_product(transa, transb, alpha, root_hmatrix, B_dense, beta, lrmat_test);
    dense_lrmat_test.resize(lrmat_test.get_U().nb_rows(), lrmat_test.get_V().nb_cols());
    lrmat_test.copy_to_dense(dense_lrmat_test.data());
    error    = normFrob(matrix_result_w_lrmat_sum - dense_lrmat_test) / normFrob(matrix_result_w_lrmat_sum);
    is_error = is_error || !(error < std::max(epsilon, lrmat_tolerance) * (1 + additional_lrmat_sum_tolerance));
    cout << "> Errors on a hmatrix matrix product to lrmat with sum: " << error << endl;

    // matrix_test = C_dense;
    // scale(scaling_coefficient, root_hmatrix);
    // add_hmatrix_matrix_product(transa, transb, alpha, root_hmatrix, B_dense, scaling_coefficient * beta, matrix_test);
    // error    = normFrob(scaled_matrix_result_w_matrix_sum - matrix_test) / normFrob(scaled_matrix_result_w_matrix_sum);
    // is_error = is_error || !(error < epsilon);
    // cout << "> Errors on a scaled hmatrix matrix product: " << error << endl;

    return is_error;
}

template <typename T, typename GeneratorTestType>
bool test_symmetric_hmatrix_matrix_product(const TestCaseSymmetricProduct<T, GeneratorTestType> &test_case, htool::underlying_type<T> epsilon, htool::underlying_type<T>) {

    int rankWorld;
    MPI_Comm_rank(MPI_COMM_WORLD, &rankWorld);
    bool is_error = false;
    double eta    = 10;
    char side     = test_case.side;
    char UPLO     = test_case.UPLO;

    const Cluster<htool::underlying_type<T>> *root_cluster_A_output, *root_cluster_A_input, *root_cluster_B_output, *root_cluster_B_input, *root_cluster_C_output, *root_cluster_C_input;
    root_cluster_A_output = &test_case.root_cluster_A_output->get_cluster_on_partition(rankWorld);
    root_cluster_A_input  = &test_case.root_cluster_A_input->get_cluster_on_partition(rankWorld);
    root_cluster_B_output = &test_case.root_cluster_B_output->get_cluster_on_partition(rankWorld);
    root_cluster_B_input  = &test_case.root_cluster_B_input->get_cluster_on_partition(rankWorld);
    root_cluster_C_output = &test_case.root_cluster_C_output->get_cluster_on_partition(rankWorld);
    root_cluster_C_input  = &test_case.root_cluster_C_input->get_cluster_on_partition(rankWorld);

    HMatrixTreeBuilder<T, htool::underlying_type<T>> hmatrix_tree_builder(epsilon, eta, 'S', UPLO, -1, std::make_shared<SVD<T>>(*test_case.operator_A));

    // build
    HMatrix<T, htool::underlying_type<T>> root_hmatrix = hmatrix_tree_builder.build(*test_case.operator_A, *root_cluster_A_output, *root_cluster_A_input);

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
    Matrix<T> matrix_test, dense_lrmat_test, transposed_B_dense(B_dense.nb_cols(), B_dense.nb_rows()), transposed_C_dense(C_dense.nb_cols(), C_dense.nb_rows());
    transpose(B_dense, transposed_B_dense);
    transpose(C_dense, transposed_C_dense);

    Matrix<T> HA_dense(root_hmatrix.get_target_cluster().get_size(), root_hmatrix.get_source_cluster().get_size());
    copy_to_dense(root_hmatrix, HA_dense.data());

    // lrmat
    SVD<T> compressor(*test_case.operator_C);
    htool::underlying_type<T> lrmat_tolerance = 0.0001;
    LowRankMatrix<T> C_auto_approximation(root_cluster_C_output->get_size(), root_cluster_C_input->get_size(), lrmat_tolerance), lrmat_test(root_cluster_C_output->get_size(), root_cluster_C_input->get_size(), lrmat_tolerance);
    compressor.copy_low_rank_approximation(root_cluster_C_output->get_size(), root_cluster_C_input->get_size(), root_cluster_C_output->get_offset(), root_cluster_C_input->get_offset(), C_auto_approximation);

    // Random Input matrix
    std::vector<T> B_vec, C_vec, test_vec;
    B_vec = get_col(B_dense, 0);
    C_vec = get_col(C_dense, 0);
    T alpha(0), beta(0), scaling_coefficient;
    htool::underlying_type<T> error;
    generate_random_scalar(alpha);
    generate_random_scalar(beta);
    generate_random_scalar(scaling_coefficient);

    // References
    Matrix<T> matrix_result_w_matrix_sum(C_dense), transposed_matrix_result_w_matrix_sum(C_dense.nb_cols(), C_dense.nb_rows()), matrix_result_wo_sum(C_dense), matrix_result_w_lrmat_sum(C_dense);
    C_auto_approximation.copy_to_dense(matrix_result_w_lrmat_sum.data());

    // if (is_complex<T>()) {
    add_symmetric_matrix_matrix_product(side, UPLO, alpha, A_dense, B_dense, beta, matrix_result_w_matrix_sum);
    transpose(matrix_result_w_matrix_sum, transposed_matrix_result_w_matrix_sum);
    // add_symmetric_matrix_matrix_product(side, UPLO, alpha, A_dense, B_dense, T(0), matrix_result_wo_sum);
    // add_symmetric_matrix_matrix_product(side, UPLO, alpha, HA_dense, B_dense, beta, matrix_result_w_lrmat_sum);
    // } else {
    //     add_hermitian_matrix_matrix_product(side, UPLO, alpha, A_dense, B_dense, beta, matrix_result_w_matrix_sum);
    //     transpose(matrix_result_w_matrix_sum, transposed_matrix_result_w_matrix_sum);
    // }

    Matrix<T> scaled_matrix_result_w_matrix_sum(matrix_result_w_matrix_sum);
    scale(scaling_coefficient, scaled_matrix_result_w_matrix_sum);

    // Product
    // if ((side == 'L' && ni_B == 1) || (side == 'R' && no_B == 1)) {
    //     test_vec = C_vec;
    //     add_symmetric_hmatrix_vector_product(side, alpha, root_hmatrix, B_vec.data(), beta, test_vec.data());
    //     error    = norm2(matrix_result_w_matrix_sum.get_col(0) - test_vec) / norm2(matrix_result_w_matrix_sum.get_col(0));
    //     is_error = is_error || !(error < epsilon);
    //     cout << "> Errors on a symmetric hmatrix vector product: " << error << endl;
    // }

    if (side == 'L') {
        matrix_test = C_dense;
        internal_add_hmatrix_matrix_product('N', 'N', alpha, root_hmatrix, B_dense, beta, matrix_test);
        error    = normFrob(matrix_result_w_matrix_sum - matrix_test) / normFrob(matrix_result_w_matrix_sum);
        is_error = is_error || !(error < epsilon);
        cout << "> Errors on a symmetric hmatrix matrix product: " << error << endl;

        matrix_test = transposed_C_dense;
        sequential_internal_add_hmatrix_matrix_product_row_major('N', 'N', alpha, root_hmatrix, transposed_B_dense.data(), beta, matrix_test.data(), C_dense.nb_cols());
        error    = normFrob(transposed_matrix_result_w_matrix_sum - matrix_test) / normFrob(transposed_matrix_result_w_matrix_sum);
        is_error = is_error || !(error < epsilon);
        cout << "> Errors on a sequential symmetric hmatrix matrix product with row major input: " << error << endl;

        matrix_test = transposed_C_dense;
        openmp_internal_add_hmatrix_matrix_product_row_major('N', 'N', alpha, root_hmatrix, transposed_B_dense.data(), beta, matrix_test.data(), C_dense.nb_cols());
        error    = normFrob(transposed_matrix_result_w_matrix_sum - matrix_test) / normFrob(transposed_matrix_result_w_matrix_sum);
        is_error = is_error || !(error < epsilon);
        cout << "> Errors on a openmp symmetric hmatrix matrix product with row major input: " << error << endl;
    } else {
        matrix_test = C_dense;
        internal_add_matrix_hmatrix_product('N', 'N', alpha, B_dense, root_hmatrix, beta, matrix_test);
        error    = normFrob(matrix_result_w_matrix_sum - matrix_test) / normFrob(matrix_result_w_matrix_sum);
        is_error = is_error || !(error < epsilon);
        cout << "> Errors on a matrix symmetric hmatrix product: " << error << endl;
    }

    return is_error;
}

template <typename T, typename GeneratorTestType>
bool test_hermitian_hmatrix_matrix_product(const TestCaseSymmetricProduct<T, GeneratorTestType> &test_case, htool::underlying_type<T> epsilon, htool::underlying_type<T>) {

    int rankWorld;
    MPI_Comm_rank(MPI_COMM_WORLD, &rankWorld);
    bool is_error = false;
    double eta    = 10;
    char side     = test_case.side;
    char UPLO     = test_case.UPLO;

    const Cluster<htool::underlying_type<T>> *root_cluster_A_output, *root_cluster_A_input, *root_cluster_B_output, *root_cluster_B_input, *root_cluster_C_output, *root_cluster_C_input;
    root_cluster_A_output = &test_case.root_cluster_A_output->get_cluster_on_partition(rankWorld);
    root_cluster_A_input  = &test_case.root_cluster_A_input->get_cluster_on_partition(rankWorld);
    root_cluster_B_output = &test_case.root_cluster_B_output->get_cluster_on_partition(rankWorld);
    root_cluster_B_input  = &test_case.root_cluster_B_input->get_cluster_on_partition(rankWorld);
    root_cluster_C_output = &test_case.root_cluster_C_output->get_cluster_on_partition(rankWorld);
    root_cluster_C_input  = &test_case.root_cluster_C_input->get_cluster_on_partition(rankWorld);

    HMatrixTreeBuilder<T, htool::underlying_type<T>> hmatrix_tree_builder(epsilon, eta, 'H', UPLO, -1, std::make_shared<SVD<T>>(*test_case.operator_A));

    // build
    HMatrix<T, htool::underlying_type<T>> root_hmatrix = hmatrix_tree_builder.build(*test_case.operator_A, *root_cluster_A_output, *root_cluster_A_input);

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
    Matrix<T> matrix_test, dense_lrmat_test, transposed_B_dense(B_dense.nb_cols(), B_dense.nb_rows()), transposed_C_dense(C_dense.nb_cols(), C_dense.nb_rows());
    transpose(B_dense, transposed_B_dense);
    transpose(C_dense, transposed_C_dense);

    Matrix<T> HA_dense(root_hmatrix.get_target_cluster().get_size(), root_hmatrix.get_source_cluster().get_size());
    copy_to_dense(root_hmatrix, HA_dense.data());

    // // lrmat
    // SVD<T> compressor(*test_case.operator_C);
    // htool::underlying_type<T> lrmat_tolerance = 0.0001;
    // LowRankMatrix<T> C_auto_approximation( compressor, *root_cluster_C_output, *root_cluster_C_input, -1, lrmat_tolerance), lrmat_test(lrmat_tolerance);

    // Random Input matrix
    std::vector<T> B_vec, C_vec, test_vec;
    B_vec = get_col(B_dense, 0);
    C_vec = get_col(C_dense, 0);
    T alpha(0), beta(0), scaling_coefficient;
    htool::underlying_type<T> error;
    generate_random_scalar(alpha);
    generate_random_scalar(beta);
    generate_random_scalar(scaling_coefficient);

    // References
    Matrix<T> matrix_result_w_matrix_sum(C_dense), transposed_matrix_result_w_matrix_sum(C_dense.nb_cols(), C_dense.nb_rows()), matrix_result_wo_sum(C_dense), matrix_result_w_lrmat_sum(C_dense);
    // C_auto_approximation.copy_to_dense(matrix_result_w_lrmat_sum.data());

    // if (is_complex<T>()) {
    add_hermitian_matrix_matrix_product(side, UPLO, alpha, A_dense, B_dense, beta, matrix_result_w_matrix_sum);
    transpose(matrix_result_w_matrix_sum, transposed_matrix_result_w_matrix_sum);
    // add_symmetric_matrix_matrix_product(side, UPLO, alpha, A_dense, B_dense, T(0), matrix_result_wo_sum);
    // add_symmetric_matrix_matrix_product(side, UPLO, alpha, HA_dense, B_dense, beta, matrix_result_w_lrmat_sum);
    // } else {
    //     add_hermitian_matrix_matrix_product(side, UPLO, alpha, A_dense, B_dense, beta, matrix_result_w_matrix_sum);
    //     transpose(matrix_result_w_matrix_sum, transposed_matrix_result_w_matrix_sum);
    // }

    Matrix<T> scaled_matrix_result_w_matrix_sum(matrix_result_w_matrix_sum);
    scale(scaling_coefficient, scaled_matrix_result_w_matrix_sum);

    // Product
    // if ((side == 'L' && ni_B == 1) || (side == 'R' && no_B == 1)) {
    //     test_vec = C_vec;
    //     add_symmetric_hmatrix_vector_product(side, alpha, root_hmatrix, B_vec.data(), beta, test_vec.data());
    //     error    = norm2(matrix_result_w_matrix_sum.get_col(0) - test_vec) / norm2(matrix_result_w_matrix_sum.get_col(0));
    //     is_error = is_error || !(error < epsilon);
    //     cout << "> Errors on a symmetric hmatrix vector product: " << error << endl;
    // }

    if (side == 'L') {
        matrix_test = C_dense;
        internal_add_hmatrix_matrix_product('N', 'N', alpha, root_hmatrix, B_dense, beta, matrix_test);
        error    = normFrob(matrix_result_w_matrix_sum - matrix_test) / normFrob(matrix_result_w_matrix_sum);
        is_error = is_error || !(error < epsilon);
        cout << "> Errors on a hermitian hmatrix matrix product: " << error << endl;

        matrix_test = transposed_C_dense;
        sequential_internal_add_hmatrix_matrix_product_row_major('N', 'N', alpha, root_hmatrix, transposed_B_dense.data(), beta, matrix_test.data(), C_dense.nb_cols());
        error    = normFrob(transposed_matrix_result_w_matrix_sum - matrix_test) / normFrob(transposed_matrix_result_w_matrix_sum);
        is_error = is_error || !(error < epsilon);
        cout << "> Errors on a sequential hermitian hmatrix matrix product with row major input: " << error << endl;

        matrix_test = transposed_C_dense;
        openmp_internal_add_hmatrix_matrix_product_row_major('N', 'N', alpha, root_hmatrix, transposed_B_dense.data(), beta, matrix_test.data(), C_dense.nb_cols());
        error    = normFrob(transposed_matrix_result_w_matrix_sum - matrix_test) / normFrob(transposed_matrix_result_w_matrix_sum);
        is_error = is_error || !(error < epsilon);
        cout << "> Errors on a openmp hermitian hmatrix matrix product with row major input: " << error << endl;
    } else {
        matrix_test = C_dense;
        // std::cout << "??? " << root_hmatrix.get_symmetry_for_leaves() << "\n";
        internal_add_matrix_hmatrix_product('N', 'N', alpha, B_dense, root_hmatrix, beta, matrix_test);
        error    = normFrob(matrix_result_w_matrix_sum - matrix_test) / normFrob(matrix_result_w_matrix_sum);
        is_error = is_error || !(error < epsilon);
        cout << "> Errors on a matrix hermitian hmatrix product: " << error << endl;
    }

    return is_error;
}
