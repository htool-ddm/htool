#include "htool/hmatrix/execution_policies.hpp"
#include "htool/matrix/linalg/factorization.hpp"
#include <htool/hmatrix/hmatrix.hpp> // for HMatrix
#include <htool/hmatrix/linalg/factorization.hpp>
#include <htool/hmatrix/tree_builder/tree_builder.hpp>       // for HMatrix...
#include <htool/matrix/linalg/add_matrix_matrix_product.hpp> // for add_her...
#include <htool/matrix/matrix.hpp>                           // for Matrix
#include <htool/misc/misc.hpp>                               // for underly...
#include <htool/testing/generate_test_case.hpp>              // for TestCas...
#include <htool/testing/generator_input.hpp>                 // for generat...
#include <iostream>                                          // for operator<<
#include <type_traits>                                       // for enable_...

using namespace std;
using namespace htool;

template <typename T, typename GeneratorTestType>
bool test_task_based_hmatrix_full_lu(char trans, int n1, int n2, htool::underlying_type<T> epsilon, htool::underlying_type<T> margin) {
    bool is_error = false;
    double eta    = 100;
    htool::underlying_type<T> error;

    // Setup test case
    htool::TestCaseSolve<T, GeneratorTestType> test_case('L', trans, n1, n2, 1, -1);

    // Policy
    omp_task_policy<T, double> policy;

    // HMatrix
    HMatrixTreeBuilder<T, htool::underlying_type<T>> hmatrix_tree_builder_A(epsilon, eta, 'N', 'N');

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

    std::unique_ptr<HMatrix<T, htool::underlying_type<T>>> A;
#if defined(_OPENMP)
#    pragma omp parallel
#    pragma omp single
#endif
    {
        // Assembly
        A = std::make_unique<HMatrix<T, htool::underlying_type<T>>>(hmatrix_tree_builder_A.build(policy, *test_case.operator_in_user_numbering_A, *test_case.root_cluster_A_output, *test_case.root_cluster_A_input));

        // Factorization
        lu_factorization(policy, *A);
    }

    // LU factorization
    matrix_test = B_dense;
    lu_solve(trans, *A, matrix_test);
    error    = normFrob(X_dense - matrix_test) / normFrob(X_dense);
    is_error = is_error || !(error < epsilon * margin);
    cout << "> Errors on hmatrix lu solve: " << error << endl;
    cout << "> is_error: " << is_error << "\n";

    return is_error;
}

template <typename T, typename GeneratorTestType>
bool test_task_based_hmatrix_full_cholesky(char UPLO, int n1, int n2, htool::underlying_type<T> epsilon, htool::underlying_type<T> margin) {
    bool is_error = false;
    double eta    = 100;
    htool::underlying_type<T> error;

    // Setup test case
    htool::TestCaseSolve<T, GeneratorTestType> test_case('L', 'N', n1, n2, 1, -1);

    // Policy
    omp_task_policy<T, double> policy;

    // HMatrix
    HMatrixTreeBuilder<T, htool::underlying_type<T>> hmatrix_tree_builder_A(epsilon, eta, is_complex<T>() ? 'H' : 'S', UPLO);

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
    if constexpr (is_complex<T>()) {
        add_hermitian_matrix_matrix_product('L', UPLO, T(1.), A_dense, X_dense, T(0.), B_dense);
    } else {
        add_symmetric_matrix_matrix_product('L', UPLO, T(1.), A_dense, X_dense, T(0.), B_dense);
    }

    std::unique_ptr<HMatrix<T, htool::underlying_type<T>>> A;
#if defined(_OPENMP)
#    pragma omp parallel
#    pragma omp single
#endif
    {
        // Assembly
        A = std::make_unique<HMatrix<T, htool::underlying_type<T>>>(hmatrix_tree_builder_A.build(*test_case.operator_in_user_numbering_A, *test_case.root_cluster_A_output, *test_case.root_cluster_A_input));

        // Factorization
        cholesky_factorization(policy, UPLO, *A);
    }

    // Cholesky factorization
    matrix_test = B_dense;
    cholesky_solve(UPLO, *A, matrix_test);
    error    = normFrob(X_dense - matrix_test) / normFrob(X_dense);
    is_error = is_error || !(error < epsilon * margin);
    cout << "> Errors on hmatrix cholesky solve: " << error << endl;
    cout << "> is_error: " << is_error << "\n";

    return is_error;
}
