#include <cmath>                             // for pow, sqrt
#include <cstddef>                           // for size_t
#include <htool/basic_types/vector.hpp>      // for norm2
#include <htool/clustering/cluster_node.hpp> // for Cluster...
#include <htool/clustering/cluster_output.hpp>
#include <htool/clustering/tree_builder/tree_builder.hpp>       // for Cluster...
#include <htool/hmatrix/hmatrix.hpp>                            // for copy_di...
#include <htool/hmatrix/hmatrix_distributed_output.hpp>         // for print_d...
#include <htool/hmatrix/hmatrix_output.hpp>                     // for print_h...
#include <htool/hmatrix/hmatrix_output_dot.hpp>                 // for view_block_tree...
#include <htool/hmatrix/interfaces/virtual_generator.hpp>       // for Generat...
#include <htool/hmatrix/linalg/add_hmatrix_hmatrix_product.hpp> // for add_...
#include <htool/hmatrix/linalg/add_hmatrix_vector_product.hpp>
#include <htool/hmatrix/linalg/factorization.hpp>
#include <htool/hmatrix/linalg/task_based_add_hmatrix_hmatrix_product.hpp>      // for task_bas...
#include <htool/hmatrix/linalg/task_based_add_hmatrix_vector_product.hpp>       // for task_bas...
#include <htool/hmatrix/linalg/task_based_factorization.hpp>                    // for task_based_lu_factorization
#include <htool/hmatrix/linalg/task_based_triangular_hmatrix_hmatrix_solve.hpp> // for task_based_triangular_hmatrix_hmatrix_solve
#include <htool/hmatrix/linalg/triangular_hmatrix_hmatrix_solve.hpp>            // for triangular_hmatrix_hmatrix_solve

#include <htool/hmatrix/lrmat/SVD.hpp>
#include <htool/hmatrix/tree_builder/task_based_tree_builder.hpp> // for enumerate_dependence, find_l0...
#include <htool/hmatrix/tree_builder/tree_builder.hpp>            // for HMatrix...
#include <htool/matrix/matrix.hpp>                                // for Matrix
#include <htool/misc/misc.hpp>                                    // for underly...
#include <htool/misc/user.hpp>                                    // for NbrToStr
#include <htool/testing/dense_blocks_generator_test.hpp>
#include <htool/testing/generate_test_case.hpp> // for TestCaseSymmetricPro...
#include <htool/testing/generator_input.hpp>
#include <htool/testing/geometry.hpp>  // for create_...
#include <htool/testing/partition.hpp> // for test_pa...
#include <iostream>                    // for operator<<
#include <memory>                      // for make_sh...
#include <string>                      // for operator+
#include <vector>                      // for vector

using namespace std;
using namespace htool;

template <typename T, typename GeneratorTestType>
bool test_task_based_lu_factorization(char trans, int n1, int n2, htool::underlying_type<T> epsilon) {

    bool is_error = false;
    double eta    = 100;
    htool::underlying_type<T> error;
    std::cout << "eta = " << eta << std::endl;
    std::cout << "task_based_lu_factorization tests...\n";

    // Setup test case
    htool::TestCaseSolve<T, GeneratorTestType> test_case('L', trans, n1, n2, 1, -1);

    // HMatrix
    HMatrixTreeBuilder<T, htool::underlying_type<T>> hmatrix_tree_builder_A(epsilon, eta, 'N', 'N');
    HMatrix<T, htool::underlying_type<T>> A = hmatrix_tree_builder_A.build(*test_case.operator_A, *test_case.root_cluster_A_output, *test_case.root_cluster_A_input);

    // Matrix
    int ni_A = test_case.root_cluster_A_input->get_size();
    int no_A = test_case.root_cluster_A_output->get_size();
    int ni_X = test_case.root_cluster_X_input->get_size();
    int no_X = test_case.root_cluster_X_output->get_size();
    Matrix<T> A_dense(no_A, ni_A), X_dense(no_X, ni_X), B_dense(X_dense), densified_hmatrix_test(B_dense), matrix_test;
    std::vector<int> identity(ni_A);
    std::iota(identity.begin(), identity.end(), test_case.root_cluster_A_output->get_offset());
    test_case.operator_in_user_numbering_A->copy_submatrix(no_A, ni_A, identity.data(), identity.data(), A_dense.data());
    generate_random_matrix(X_dense);
    add_matrix_matrix_product(trans, 'N', T(1.), A_dense, X_dense, T(0.), B_dense);

    // Tests
    std::chrono::steady_clock::time_point start, end;

    //// Classic LU factorization
    auto A_classic = A;
    matrix_test    = B_dense;
    start          = std::chrono::steady_clock::now();
    lu_factorization(A_classic);
    end = std::chrono::steady_clock::now();
    // save_leaves_with_rank(A_classic, "classic_hmatrix_facto");
    lu_solve(trans, A_classic, matrix_test);
    std::chrono::duration<double> classic_duration = end - start;
    error                                          = normFrob(X_dense - matrix_test) / normFrob(X_dense);
    is_error                                       = is_error || !(error < epsilon);
    cout << ">   classic error = " << error << endl;
    cout << "    classic_duration = " << classic_duration.count() << std::endl;

    //// Task-based LU factorization
    auto A_task_based              = A;
    int max_nb_nodes               = 64;
    std::vector<HMatrix<T> *> L0_A = find_l0(A_task_based, max_nb_nodes);
    matrix_test                    = B_dense;
    start                          = std::chrono::steady_clock::now();
#if defined(_OPENMP) && !defined(HTOOL_WITH_PYTHON_INTERFACE)
#    pragma omp parallel
#    pragma omp single
#endif
    {
        task_based_lu_factorization(A_task_based, L0_A);
    }
    end = std::chrono::steady_clock::now();
    // save_leaves_with_rank(A_task_based, "TB_hmatrix_facto");
    lu_solve(trans, A_task_based, matrix_test);
    std::chrono::duration<double> task_based_duration = end - start;
    error                                             = normFrob(X_dense - matrix_test) / normFrob(X_dense);
    is_error                                          = is_error || !(error < epsilon);
    cout << ">   task_based error = " << error << endl;
    cout << "    task_based_duration = " << task_based_duration.count() << std::endl;
    if (task_based_duration.count() > classic_duration.count()) {
        htool::Logger::get_instance().log(LogLevel::ERROR, "Careful: task_based_duration > classic_duration. Ratio TB/Classic = " + std::to_string(task_based_duration.count() / classic_duration.count()) + "."); // LCOV_EXCL_LINE
    }

    // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // Print the results
    // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if (is_error) {
        htool::Logger::get_instance().log(LogLevel::ERROR, "test_task_based_lu_factorization current case failed."); // LCOV_EXCL_LINE
    } else {
        std::cout << "SUCCESS: test_task_based_lu_factorization current case passed." << std::endl;
        std::cout << "===============================================================\n"
                  << std::endl;
    }
    return is_error;
}

template <typename T, typename GeneratorTestType, std::enable_if_t<!is_complex_t<T>::value, bool> = true>
bool test_task_based_cholesky_factorization(char UPLO, int n1, int n2, htool::underlying_type<T> epsilon) {
    bool is_error = false;
    double eta    = 100;
    htool::underlying_type<T> error;
    std::cout << "eta = " << eta << std::endl;
    std::cout << "task_based_cholesky_factorization tests...\n";

    // Setup test case
    htool::TestCaseSolve<T, GeneratorTestType> test_case('L', 'N', n1, n2, 1, -1);

    // HMatrix
    HMatrixTreeBuilder<T, htool::underlying_type<T>> hmatrix_tree_builder_A(epsilon, eta, is_complex<T>() ? 'H' : 'S', UPLO);
    HMatrix<T, htool::underlying_type<T>> HA = hmatrix_tree_builder_A.build(*test_case.operator_in_user_numbering_A, *test_case.root_cluster_A_output, *test_case.root_cluster_A_input);

    // Matrix
    int ni_A = test_case.root_cluster_A_input->get_size();
    int no_A = test_case.root_cluster_A_output->get_size();
    int ni_X = test_case.root_cluster_X_input->get_size();
    int no_X = test_case.root_cluster_X_output->get_size();
    Matrix<T> A_dense(no_A, ni_A), X_dense(no_X, ni_X), B_dense(X_dense), densified_hmatrix_test(B_dense), matrix_test;
    std::vector<int> identity(ni_A);
    std::iota(identity.begin(), identity.end(), test_case.root_cluster_A_output->get_offset());
    test_case.operator_in_user_numbering_A->copy_submatrix(no_A, ni_A, identity.data(), identity.data(), A_dense.data());
    generate_random_matrix(X_dense);
    add_symmetric_matrix_matrix_product('L', UPLO, T(1.), A_dense, X_dense, T(0.), B_dense);

    // Tests
    std::chrono::steady_clock::time_point start, end;
    auto task_based_HA = HA;

    //// Classic Cholesky factorization
    matrix_test = B_dense;
    start       = std::chrono::steady_clock::now();
    cholesky_factorization(UPLO, HA);
    end = std::chrono::steady_clock::now();
    cholesky_solve(UPLO, HA, matrix_test);

    std::chrono::duration<double> classic_duration = end - start;
    error                                          = normFrob(X_dense - matrix_test) / normFrob(X_dense);
    is_error                                       = is_error || !(error < epsilon);
    cout << ">   classic error = " << error << endl;
    cout << "    classic_duration = " << classic_duration.count() << std::endl;

    //// Task-based Cholesky factorization
    int max_nb_nodes               = 64;
    std::vector<HMatrix<T> *> L0_A = find_l0(task_based_HA, max_nb_nodes);
    matrix_test                    = B_dense;
    start                          = std::chrono::steady_clock::now();
#if defined(_OPENMP) && !defined(HTOOL_WITH_PYTHON_INTERFACE)
#    pragma omp parallel
#    pragma omp single
#endif
    {
        task_based_cholesky_factorization(UPLO, task_based_HA, L0_A);
    }
    end = std::chrono::steady_clock::now();
    cholesky_solve(UPLO, task_based_HA, matrix_test);

    std::chrono::duration<double> task_based_duration = end - start;
    error                                             = normFrob(X_dense - matrix_test) / normFrob(X_dense);
    is_error                                          = is_error || !(error < epsilon);
    cout << ">   task_based error = " << error << endl;
    cout << "    task_based_duration = " << task_based_duration.count() << std::endl;
    if (task_based_duration.count() > classic_duration.count()) {
        htool::Logger::get_instance().log(LogLevel::ERROR, "Careful: task_based_duration > classic_duration. Ratio TB/Classic = " + std::to_string(task_based_duration.count() / classic_duration.count()) + "."); // LCOV_EXCL_LINE
    }

    // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // Print the results
    // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if (is_error) {
        htool::Logger::get_instance().log(LogLevel::ERROR, "test_task_based_cholesky_factorization current case failed."); // LCOV_EXCL_LINE
    } else {
        std::cout << "SUCCESS: test_task_based_cholesky_factorization current case passed." << std::endl;
        std::cout << "===============================================================\n"
                  << std::endl;
    }
    return is_error;
} // end of test_task_based_cholesky_factorization

template <typename T, typename GeneratorTestType, std::enable_if_t<is_complex_t<T>::value, bool> = true>
bool test_task_based_cholesky_factorization(char UPLO, int n1, int n2, htool::underlying_type<T> epsilon) {
    bool is_error = false;
    double eta    = 100;
    htool::underlying_type<T> error;
    std::cout << "eta = " << eta << std::endl;
    std::cout << "task_based_cholesky_factorization tests...\n";

    // Setup test case
    htool::TestCaseSolve<T, GeneratorTestType> test_case('L', 'N', n1, n2, 1, -1);

    // HMatrix
    HMatrixTreeBuilder<T, htool::underlying_type<T>> hmatrix_tree_builder_A(epsilon, eta, is_complex<T>() ? 'H' : 'S', UPLO);
    HMatrix<T, htool::underlying_type<T>> HA = hmatrix_tree_builder_A.build(*test_case.operator_in_user_numbering_A, *test_case.root_cluster_A_output, *test_case.root_cluster_A_input);

    // Matrix
    int ni_A = test_case.root_cluster_A_input->get_size();
    int no_A = test_case.root_cluster_A_output->get_size();
    int ni_X = test_case.root_cluster_X_input->get_size();
    int no_X = test_case.root_cluster_X_output->get_size();
    Matrix<T> A_dense(no_A, ni_A), X_dense(no_X, ni_X), B_dense(X_dense), densified_hmatrix_test(B_dense), matrix_test;
    std::vector<int> identity(ni_A);
    std::iota(identity.begin(), identity.end(), test_case.root_cluster_A_output->get_offset());
    test_case.operator_in_user_numbering_A->copy_submatrix(no_A, ni_A, identity.data(), identity.data(), A_dense.data());
    generate_random_matrix(X_dense);
    add_hermitian_matrix_matrix_product('L', UPLO, T(1.), A_dense, X_dense, T(0.), B_dense);

    // Tests
    std::chrono::steady_clock::time_point start, end;
    auto task_based_HA = HA;

    //// Classic Cholesky factorization
    matrix_test = B_dense;
    start       = std::chrono::steady_clock::now();
    cholesky_factorization(UPLO, HA);
    end = std::chrono::steady_clock::now();
    cholesky_solve(UPLO, HA, matrix_test);
    std::chrono::duration<double> classic_duration = end - start;

    error    = normFrob(X_dense - matrix_test) / normFrob(X_dense);
    is_error = is_error || !(error < epsilon);
    cout << ">   classic error = " << error << endl;
    cout << "    classic_duration = " << classic_duration.count() << std::endl;

    //// Task-based Cholesky factorization
    int max_nb_nodes               = 64;
    std::vector<HMatrix<T> *> L0_A = find_l0(task_based_HA, max_nb_nodes);
    matrix_test                    = B_dense;
    start                          = std::chrono::steady_clock::now();
#if defined(_OPENMP) && !defined(HTOOL_WITH_PYTHON_INTERFACE)
#    pragma omp parallel
#    pragma omp single
#endif
    {
        task_based_cholesky_factorization(UPLO, task_based_HA, L0_A);
    }
    end = std::chrono::steady_clock::now();
    cholesky_solve(UPLO, task_based_HA, matrix_test);

    std::chrono::duration<double> task_based_duration = end - start;
    error                                             = normFrob(X_dense - matrix_test) / normFrob(X_dense);
    is_error                                          = is_error || !(error < epsilon);
    cout << ">   task_based error = " << error << endl;
    cout << "    task_based_duration = " << task_based_duration.count() << std::endl;
    if (task_based_duration.count() > classic_duration.count()) {
        htool::Logger::get_instance().log(LogLevel::ERROR, "Careful: task_based_duration > classic_duration. Ratio TB/Classic = " + std::to_string(task_based_duration.count() / classic_duration.count()) + "."); // LCOV_EXCL_LINE
    }

    // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // Print the results
    // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if (is_error) {
        htool::Logger::get_instance().log(LogLevel::ERROR, "test_task_based_cholesky_factorization current case failed."); // LCOV_EXCL_LINE
    } else {
        std::cout << "SUCCESS: test_task_based_cholesky_factorization current case passed." << std::endl;
        std::cout << "===============================================================\n"
                  << std::endl;
    }
    return is_error;
}
