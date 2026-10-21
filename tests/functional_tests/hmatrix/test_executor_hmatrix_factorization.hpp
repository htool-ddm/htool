#include <htool/clustering/tree_builder/tree_builder.hpp> // for ClusterTreeBuilder
#include <htool/hmatrix/hmatrix.hpp>                      // for HMatrix
#include <htool/hmatrix/linalg/add_hmatrix_vector_product.hpp>
#include <htool/hmatrix/linalg/executor/factorization.hpp>
#include <htool/hmatrix/linalg/executor/lazy_factorization.hpp>
#include <htool/hmatrix/linalg/executor/thread_executor.hpp>
#include <htool/hmatrix/linalg/factorization.hpp>
#include <htool/hmatrix/tree_builder/tree_builder.hpp>       // for HMatrix...
#include <htool/matrix/linalg/add_matrix_matrix_product.hpp> // for add_matrix_matrix_product
#include <htool/matrix/matrix.hpp>                           // for Matrix
#include <htool/misc/misc.hpp>                               // for underly...
#include <htool/testing/generate_test_case.hpp>              // for TestCas...
#include <htool/testing/generator_input.hpp>                 // for generat...
#include <htool/testing/generator_test.hpp>                  // for GeneratorTest...
#include <htool/testing/single_layer_sphere.hpp>             // for SingleLayerSphere
#include <iostream>                                          // for operator<<

using namespace std;
using namespace htool;

// Largest difference between the entries of two HMatrix with the same block tree
template <typename T>
htool::underlying_type<T> max_difference(const HMatrix<T, htool::underlying_type<T>> &A, const HMatrix<T, htool::underlying_type<T>> &B) {
    Matrix<T> A_dense(A.nb_rows(), A.nb_cols()), B_dense(B.nb_rows(), B.nb_cols());
    copy_to_dense(A, A_dense.data());
    copy_to_dense(B, B_dense.data());
    htool::underlying_type<T> difference = 0;
    for (int i = 0; i < A_dense.nb_rows() * A_dense.nb_cols(); i++) {
        difference = std::max(difference, std::abs(A_dense.data()[i] - B_dense.data()[i]));
    }
    return difference;
}

// Eager LU run by a SerialExecutor and a ThreadExecutor, which must give the factors of sequential_lu_factorization, and solve with an error
// below epsilon * margin.
template <typename T, typename GeneratorTestType>
bool test_executor_hmatrix_lu(char trans, int n1, int n2, htool::underlying_type<T> epsilon, double eta, htool::underlying_type<T> margin, int number_of_threads) {
    bool is_error = false;
    htool::underlying_type<T> error;

    // Setup test case
    htool::TestCaseSolve<T, GeneratorTestType> test_case('L', trans, n1, n2, 1, -1);

    // HMatrix
    HMatrixTreeBuilder<T, htool::underlying_type<T>> hmatrix_tree_builder_A(epsilon, eta, 'N', 'N');
    HMatrix<T, htool::underlying_type<T>> A = hmatrix_tree_builder_A.build(*test_case.operator_in_user_numbering_A, *test_case.root_cluster_A_output, *test_case.root_cluster_A_input);

    // Matrix
    int ni_A = test_case.root_cluster_A_input->get_size();
    int no_A = test_case.root_cluster_A_output->get_size();
    int ni_X = test_case.root_cluster_X_input->get_size();
    int no_X = test_case.root_cluster_X_output->get_size();
    Matrix<T> A_dense(no_A, ni_A), X_dense(no_X, ni_X), B_dense(X_dense), matrix_test;
    std::vector<int> identity(ni_A);
    std::iota(identity.begin(), identity.end(), test_case.root_cluster_A_output->get_offset());
    test_case.operator_in_user_numbering_A->copy_submatrix(no_A, ni_A, identity.data(), identity.data(), A_dense.data());
    generate_random_matrix(X_dense);
    add_matrix_matrix_product(trans, 'N', T(1.), A_dense, X_dense, T(0.), B_dense);

    // LU factorization
    SerialExecutor<T> serial_executor;
    ThreadExecutor<T> thread_executor(number_of_threads, true);
    HMatrix<T, htool::underlying_type<T>> A_sequential(A), A_serial(A), A_threads(A);
    sequential_lu_factorization(A_sequential);
    executor_lu_factorization(serial_executor, A_serial);
    executor_lu_factorization(thread_executor, A_threads);
    error    = std::max(max_difference(A_sequential, A_serial), max_difference(A_sequential, A_threads));
    is_error = is_error || !(error == 0);
    cout << "> Differences between lu run by executors and sequential lu: " << error << '\n';

    matrix_test = B_dense;
    lu_solve(trans, A_threads, matrix_test);
    error    = normFrob(X_dense - matrix_test) / normFrob(X_dense);
    is_error = is_error || !(error < epsilon * margin);
    cout << "> Errors on hmatrix lu solve with lu run by a thread executor: " << error << '\n';
    cout << "> is_error: " << is_error << "\n";

    return is_error;
}

// Lazy LU run by a SerialExecutor and a ThreadExecutor, with and without recompression, which must give the same factors, and solve with an
// error below epsilon * margin.
template <typename T, typename GeneratorTestType>
bool test_lazy_hmatrix_lu(char trans, int n1, int n2, htool::underlying_type<T> epsilon, double eta, htool::underlying_type<T> margin, int number_of_threads) {
    bool is_error = false;
    htool::underlying_type<T> error;

    // Setup test case
    htool::TestCaseSolve<T, GeneratorTestType> test_case('L', trans, n1, n2, 1, -1);

    // HMatrix
    HMatrixTreeBuilder<T, htool::underlying_type<T>> hmatrix_tree_builder_A(epsilon, eta, 'N', 'N');
    HMatrix<T, htool::underlying_type<T>> A = hmatrix_tree_builder_A.build(*test_case.operator_in_user_numbering_A, *test_case.root_cluster_A_output, *test_case.root_cluster_A_input);

    // Matrix
    int ni_A = test_case.root_cluster_A_input->get_size();
    int no_A = test_case.root_cluster_A_output->get_size();
    int ni_X = test_case.root_cluster_X_input->get_size();
    int no_X = test_case.root_cluster_X_output->get_size();
    Matrix<T> A_dense(no_A, ni_A), X_dense(no_X, ni_X), B_dense(X_dense), matrix_test;
    std::vector<int> identity(ni_A);
    std::iota(identity.begin(), identity.end(), test_case.root_cluster_A_output->get_offset());
    test_case.operator_in_user_numbering_A->copy_submatrix(no_A, ni_A, identity.data(), identity.data(), A_dense.data());
    generate_random_matrix(X_dense);
    add_matrix_matrix_product(trans, 'N', T(1.), A_dense, X_dense, T(0.), B_dense);

    // Lazy LU factorization
    SerialExecutor<T> serial_executor;
    ThreadExecutor<T> thread_executor(number_of_threads, true);
    for (bool recompress : {false, true}) {
        HMatrix<T, htool::underlying_type<T>> A_serial(A), A_threads(A);
        lazy_lu_factorization(serial_executor, A_serial, recompress);
        lazy_lu_factorization(thread_executor, A_threads, recompress);
        error    = max_difference(A_serial, A_threads);
        is_error = is_error || !(error == 0);
        cout << "> Differences between lazy lu" << (recompress ? " with recompression" : "") << " run by a serial and a thread executor: " << error << '\n';

        matrix_test = B_dense;
        lu_solve(trans, A_threads, matrix_test);
        error    = normFrob(X_dense - matrix_test) / normFrob(X_dense);
        is_error = is_error || !(error < epsilon * margin);
        cout << "> Errors on hmatrix lu solve with lazy lu" << (recompress ? " with recompression" : "") << ": " << error << '\n';
    }
    cout << "> is_error: " << is_error << "\n";

    return is_error;
}

// Residual |A x - b| / |b| of the solution x of A x = b computed with the factors, for a deterministic b
template <typename T>
htool::underlying_type<T> lu_residual(const HMatrix<T, htool::underlying_type<T>> &A, const HMatrix<T, htool::underlying_type<T>> &factors) {
    int n = A.nb_rows();
    std::vector<T> b(n), r(n);
    for (int i = 0; i < n; i++) {
        b[i] = T(std::cos(0.37 * i));
    }
    Matrix<T> x(n, 1);
    std::copy(b.begin(), b.end(), x.data());
    lu_solve('N', factors, x);
    add_hmatrix_vector_product('N', T(1), A, x.data(), T(0), r.data());
    return norm2(r - b) / norm2(b);
}

// Lazy LU of the single-layer operator of the sphere (Laplace if T is real, Helmholtz if T is complex), whose LU depends on its Schur
// complements: its residual must be within a factor margin of the one of sequential_lu_factorization.
template <typename T>
bool test_lazy_hmatrix_lu_first_kind(int n, htool::underlying_type<T> epsilon, htool::underlying_type<T> margin, int number_of_threads) {
    bool is_error = false;

    std::vector<double> points = fibonacci_sphere(n);
    SingleLayerSphere<T> generator(points, is_complex<T>() ? 8. : 0.);
    ClusterTreeBuilder<htool::underlying_type<T>> cluster_tree_builder;
    cluster_tree_builder.set_maximal_leaf_size(50);
    Cluster<htool::underlying_type<T>> cluster = cluster_tree_builder.create_cluster_tree(n, 3, points.data(), 2, 2);
    HMatrixTreeBuilder<T, htool::underlying_type<T>> hmatrix_tree_builder(epsilon, 3, 'N', 'N');
    HMatrix<T, htool::underlying_type<T>> A = hmatrix_tree_builder.build(generator, cluster, cluster);

    HMatrix<T, htool::underlying_type<T>> A_eager(A);
    sequential_lu_factorization(A_eager);
    htool::underlying_type<T> eager_residual = lu_residual(A, A_eager);
    cout << "> Residual of sequential lu (first kind, n=" << n << ", epsilon=" << epsilon << "): " << eager_residual << '\n';

    ThreadExecutor<T> thread_executor(number_of_threads, true);
    for (bool recompress : {false, true}) {
        HMatrix<T, htool::underlying_type<T>> A_lazy(A);
        lazy_lu_factorization(thread_executor, A_lazy, recompress);
        htool::underlying_type<T> lazy_residual = lu_residual(A, A_lazy);
        is_error                                = is_error || !(lazy_residual < margin * eager_residual);
        cout << "> Residual of lazy lu" << (recompress ? " with recompression" : "") << ": " << lazy_residual << '\n';
    }
    cout << "> is_error: " << is_error << "\n";

    return is_error;
}
