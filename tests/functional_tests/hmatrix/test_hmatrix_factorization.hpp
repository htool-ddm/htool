#include <htool/hmatrix/hmatrix.hpp> // for HMatrix
#include <htool/hmatrix/linalg/factorization.hpp>
#include <htool/hmatrix/tree_builder/tree_builder.hpp>       // for HMatrix...
#include <htool/matrix/linalg/add_matrix_matrix_product.hpp> // for add_her...
#include <htool/matrix/matrix.hpp>                           // for Matrix
#include <htool/misc/misc.hpp>                               // for underly...
#include <htool/testing/generate_test_case.hpp>              // for TestCas...
#include <htool/testing/generator_input.hpp>                 // for generat...
#include <htool/testing/generator_test.hpp>                  // for GeneratorInUserNumberingFromMatrix
#include <iostream>                                          // for operator<<
#include <type_traits>                                       // for enable_...

using namespace std;
using namespace htool;

template <typename T, typename GeneratorTestType>
bool test_hmatrix_lu(char trans, int n1, int n2, htool::underlying_type<T> epsilon, htool::underlying_type<T> margin) {
    bool is_error = false;
    double eta    = 100;
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
    Matrix<T> A_dense(no_A, ni_A), X_dense(no_X, ni_X), B_dense(X_dense), densified_hmatrix_test(B_dense), matrix_test;
    std::vector<int> identity(ni_A);
    std::iota(identity.begin(), identity.end(), test_case.root_cluster_A_output->get_offset());
    test_case.operator_in_user_numbering_A->copy_submatrix(no_A, ni_A, identity.data(), identity.data(), A_dense.data());
    generate_random_matrix(X_dense);
    add_matrix_matrix_product(trans, 'N', T(1.), A_dense, X_dense, T(0.), B_dense);

    // LU factorization
    matrix_test = B_dense;
    sequential_lu_factorization(A);
    lu_solve(trans, A, matrix_test);
    error    = normFrob(X_dense - matrix_test) / normFrob(X_dense);
    is_error = is_error || !(error < epsilon * margin);
    cout << "> Errors on hmatrix lu solve: " << error << '\n';
    cout << "> is_error: " << is_error << "\n";

    return is_error;
}

template <typename T, typename GeneratorTestType, std::enable_if_t<!is_complex_t<T>::value, bool> = true>
bool test_hmatrix_cholesky(char UPLO, bool full_storage, int n1, int n2, htool::underlying_type<T> epsilon, htool::underlying_type<T> margin) {
    bool is_error = false;
    double eta    = 100;
    htool::underlying_type<T> error;

    // Setup test case
    htool::TestCaseSolve<T, GeneratorTestType> test_case('L', 'N', n1, n2, 1, -1);

    // HMatrix
    HMatrixTreeBuilder<T, htool::underlying_type<T>> hmatrix_tree_builder_A(epsilon, eta, full_storage ? 'N' : (is_complex<T>() ? 'H' : 'S'), full_storage ? 'N' : UPLO);
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

    // Cholesky factorization
    matrix_test = B_dense;
    sequential_cholesky_factorization(UPLO, HA);
    cholesky_solve(UPLO, HA, matrix_test);
    error    = normFrob(X_dense - matrix_test) / normFrob(X_dense);
    is_error = is_error || !(error < epsilon * margin);
    cout << "> Errors on hmatrix cholesky solve" << (full_storage ? " (full storage)" : "") << ": " << error << '\n';
    cout << "> is_error: " << is_error << "\n";

    return is_error;
}

template <typename T, typename GeneratorTestType, std::enable_if_t<is_complex_t<T>::value, bool> = true>
bool test_hmatrix_cholesky(char UPLO, bool full_storage, int n1, int n2, htool::underlying_type<T> epsilon, htool::underlying_type<T> margin) {
    bool is_error = false;
    double eta    = 100;
    htool::underlying_type<T> error;

    // Setup test case
    htool::TestCaseSolve<T, GeneratorTestType> test_case('L', 'N', n1, n2, 1, -1);

    // HMatrix
    HMatrixTreeBuilder<T, htool::underlying_type<T>> hmatrix_tree_builder_A(epsilon, eta, full_storage ? 'N' : (is_complex<T>() ? 'H' : 'S'), full_storage ? 'N' : UPLO);
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

    // Cholesky factorization
    matrix_test = B_dense;
    sequential_cholesky_factorization(UPLO, HA);
    cholesky_solve(UPLO, HA, matrix_test);
    error    = normFrob(X_dense - matrix_test) / normFrob(X_dense);
    is_error = is_error || !(error < epsilon * margin);
    cout << "> Errors on hmatrix cholesky solve" << (full_storage ? " (full storage)" : "") << ": " << error << '\n';
    cout << "> is_error: " << is_error << "\n";

    return is_error;
}

template <typename T, typename GeneratorTestType>
bool test_hmatrix_ldlt(char UPLO, char symmetry, bool full_storage, int n1, int n2, htool::underlying_type<T> epsilon, htool::underlying_type<T> margin) {
    bool is_error = false;
    double eta    = 100;
    htool::underlying_type<T> error;

    // Setup test case
    htool::TestCaseSolve<T, GeneratorTestType> test_case('L', 'N', n1, n2, 1, -1);

    // Matrix
    int ni_A = test_case.root_cluster_A_input->get_size();
    int no_A = test_case.root_cluster_A_output->get_size();
    int ni_X = test_case.root_cluster_X_input->get_size();
    int no_X = test_case.root_cluster_X_output->get_size();
    Matrix<T> A_dense(no_A, ni_A), X_dense(no_X, ni_X), B_dense(X_dense), densified_hmatrix_test(B_dense), matrix_test;
    std::vector<int> identity(ni_A);
    std::iota(identity.begin(), identity.end(), test_case.root_cluster_A_output->get_offset());
    test_case.operator_in_user_numbering_A->copy_submatrix(no_A, ni_A, identity.data(), identity.data(), A_dense.data());

    // Within each cluster leaf, pairs consecutive rows into dominant [[0,d],[conj(d),0]] blocks so that
    // Bunch-Kaufman needs 2x2 pivots; the phase is non-real so that Hermitian ones exercise conjugation.
    T omega(0.6);
    if constexpr (is_complex<T>()) {
        omega += T(0, 0.8);
    }
    const std::vector<int> &permutation = test_case.root_cluster_A_output->get_permutation();
    preorder_tree_traversal(*test_case.root_cluster_A_output, [&](const Cluster<htool::underlying_type<T>> &cluster) {
        if (cluster.is_leaf()) {
            for (int k = cluster.get_offset(); k + 1 < cluster.get_offset() + cluster.get_size(); k += 2) {
                int u = permutation[k], v = permutation[k + 1];
                T d           = A_dense(u, u) * omega;
                A_dense(u, v) = d;
                A_dense(v, u) = symmetry == 'H' ? conj_if_complex(d) : d;
                A_dense(u, u) = 0;
                A_dense(v, v) = 0;
            }
        }
    });

    // HMatrix
    GeneratorInUserNumberingFromMatrix<T> generator_A(A_dense);
    HMatrixTreeBuilder<T, htool::underlying_type<T>> hmatrix_tree_builder_A(epsilon, eta, full_storage ? 'N' : symmetry, full_storage ? 'N' : UPLO);
    HMatrix<T, htool::underlying_type<T>> HA = hmatrix_tree_builder_A.build(generator_A, *test_case.root_cluster_A_output, *test_case.root_cluster_A_input);

    generate_random_matrix(X_dense);
    if (symmetry == 'H') {
        add_hermitian_matrix_matrix_product('L', UPLO, T(1.), A_dense, X_dense, T(0.), B_dense);
    } else {
        add_symmetric_matrix_matrix_product('L', UPLO, T(1.), A_dense, X_dense, T(0.), B_dense);
    }

    // LDLt factorization
    matrix_test = B_dense;
    sequential_ldlt_factorization(symmetry, UPLO, HA);
    ldlt_solve(symmetry, UPLO, HA, matrix_test);

    error    = normFrob(X_dense - matrix_test) / normFrob(X_dense);
    is_error = is_error || !(error < epsilon * margin);
    cout << "> Errors on hmatrix " << (symmetry == 'H' ? "hermitian" : "symmetric") << " ldlt solve" << (full_storage ? " (full storage)" : "") << ": " << error << '\n';
    cout << "> is_error: " << is_error << "\n";

    return is_error;
}

// A is a single dense leaf holding the LDLt factorization of a whole diagonal block, and B is
// hierarchical: B's children have no matching diagonal block in A, so internal_apply_ldlt_diagonal
// densifies B. The diagonal is small so that Bunch-Kaufman needs 2x2 pivots.
template <typename T, typename GeneratorTestType>
bool test_hmatrix_apply_ldlt_diagonal(char symmetry, char side, char UPLO, int n1) {
    bool is_error = false;
    htool::TestCaseSolve<T, GeneratorTestType> test_case('L', 'N', n1, 10, 1, -1);
    const Cluster<htool::underlying_type<T>> &cluster = *test_case.root_cluster_A_output;
    int n                                             = cluster.get_size();

    auto A_dense = std::make_unique<Matrix<T>>(n, n);
    generate_random_matrix(*A_dense);
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < j; i++) {
            (*A_dense)(i, j) = symmetry == 'H' ? conj_if_complex((*A_dense)(j, i)) : (*A_dense)(j, i);
        }
        (*A_dense)(j, j) = T(0.01) * (symmetry == 'H' ? T(std::real((*A_dense)(j, j))) : (*A_dense)(j, j));
    }
    ldlt_factorization(symmetry, UPLO, *A_dense);
    HMatrix<T, htool::underlying_type<T>> A(cluster, cluster);
    A.set_dense_data(std::move(A_dense));

    HMatrixTreeBuilder<T, htool::underlying_type<T>> hmatrix_tree_builder(1e-6, 10, 'N', 'N');
    HMatrix<T, htool::underlying_type<T>> B = hmatrix_tree_builder.build(*test_case.operator_in_user_numbering_A, cluster, cluster);
    is_error                                = is_error || !B.is_hierarchical();

    Matrix<T> expected(n, n), result(n, n);
    copy_to_dense(B, expected.data());
    apply_ldlt_diagonal(symmetry, side, UPLO, *A.get_dense_data(), expected);

    internal_apply_ldlt_diagonal(symmetry, side, UPLO, A, B);
    copy_to_dense(B, result.data());
    htool::underlying_type<T> error = normFrob(result - expected) / normFrob(expected);
    is_error                        = is_error || !B.is_dense() || !(error < 1e-14);
    cout << "> Errors on densified hmatrix apply ldlt diagonal (" << symmetry << ", side=" << side << ", UPLO=" << UPLO << "): " << error << '\n';

    return is_error;
}
