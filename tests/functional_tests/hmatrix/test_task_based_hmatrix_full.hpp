#include "htool/hmatrix/execution_policies.hpp"
#include "htool/matrix/linalg/factorization.hpp"
#include <cmath>
#include <htool/clustering/implementations/partitioning.hpp> // for GeometricSplitting
#include <htool/clustering/tree_builder/tree_builder.hpp>    // for ClusterTreeBuilder
#include <htool/hmatrix/hmatrix.hpp>                         // for HMatrix
#include <htool/hmatrix/hmatrix_output.hpp>                  // for get_hmatrix_information
#include <htool/hmatrix/linalg/factorization.hpp>
#include <htool/hmatrix/tree_builder/tree_builder.hpp>       // for HMatrix...
#include <htool/matrix/linalg/add_matrix_matrix_product.hpp> // for add_her...
#include <htool/matrix/matrix.hpp>                           // for Matrix
#include <htool/misc/misc.hpp>                               // for underly...
#include <htool/testing/generate_test_case.hpp>              // for TestCas...
#include <htool/testing/generator_input.hpp>
#include <htool/testing/generator_test.hpp> // for generat...
#include <iostream>                         // for operator<<
#include <random>
#include <string>
#include <type_traits> // for enable_...

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
    cout << "> Errors on hmatrix lu solve: " << error << '\n';
    cout << "> is_error: " << is_error << "\n";

    // Policy reused for another matrix: L0 of A, cached in policy, must not be used for A_bis
    HMatrix<T, htool::underlying_type<T>> A_bis = hmatrix_tree_builder_A.build(*test_case.operator_in_user_numbering_A, *test_case.root_cluster_A_output, *test_case.root_cluster_A_input);
    is_error                                    = is_error || !policy.hmatrix_task_dependencies.is_L0_of(*A) || policy.hmatrix_task_dependencies.is_L0_of(A_bis);
    lu_factorization(policy, A_bis);
    is_error    = is_error || !policy.hmatrix_task_dependencies.is_L0_of(A_bis);
    matrix_test = B_dense;
    lu_solve(trans, A_bis, matrix_test);
    error    = normFrob(X_dense - matrix_test) / normFrob(X_dense);
    is_error = is_error || !(error < epsilon * margin);
    cout << "> Errors on hmatrix lu solve with reused policy: " << error << '\n';
    cout << "> is_error: " << is_error << "\n";

    // Two builds in a row with the same builder and policy, without waiting for the tasks of the first one
    std::unique_ptr<HMatrix<T, htool::underlying_type<T>>> A_first, A_second;
#if defined(_OPENMP)
#    pragma omp parallel
#    pragma omp single
#endif
    {
        A_first  = std::make_unique<HMatrix<T, htool::underlying_type<T>>>(hmatrix_tree_builder_A.build(policy, *test_case.operator_in_user_numbering_A, *test_case.root_cluster_A_output, *test_case.root_cluster_A_input));
        A_second = std::make_unique<HMatrix<T, htool::underlying_type<T>>>(hmatrix_tree_builder_A.build(policy, *test_case.operator_in_user_numbering_A, *test_case.root_cluster_A_output, *test_case.root_cluster_A_input));
    }
    for (auto &hmatrix : {A_first.get(), A_second.get()}) {
        Matrix<T> densified_hmatrix(no_A, ni_A);
        copy_to_dense_in_user_numbering(*hmatrix, densified_hmatrix.data());
        error    = normFrob(A_dense - densified_hmatrix) / normFrob(A_dense);
        is_error = is_error || !(error < epsilon * margin);
        cout << "> Errors on hmatrix built twice in a row: " << error << '\n';
    }
    // Number of false positives, computed from the HMatrix information: the same as for a sequential build
    HMatrix<T, htool::underlying_type<T>> A_reference = hmatrix_tree_builder_A.build(*test_case.operator_in_user_numbering_A, *test_case.root_cluster_A_output, *test_case.root_cluster_A_input);
    auto information_reference                        = get_hmatrix_information(A_reference);
    for (auto &hmatrix : {A_first.get(), A_second.get()}) {
        auto information = get_hmatrix_information(*hmatrix);
        is_error         = is_error || information.count("Number_of_false_positive") == 0 || information["Number_of_false_positive"] != information_reference["Number_of_false_positive"] || information["Number_of_admissible_blocks"] != information_reference["Number_of_admissible_blocks"];
        cout << "> Number of false positives of hmatrix built twice in a row: " << information["Number_of_false_positive"] << " (" << information_reference["Number_of_false_positive"] << " for a sequential build, out of " << information_reference["Number_of_admissible_blocks"] << " admissible blocks)\n";
    }

    // L0 reduced to the root: build and factorization run sequentially
    omp_task_policy<T, double> policy_root;
    policy_root.hmatrix_task_dependencies.max_number_of_nodes = 1;
    std::unique_ptr<HMatrix<T, htool::underlying_type<T>>> A_root;
#if defined(_OPENMP)
#    pragma omp parallel
#    pragma omp single
#endif
    {
        A_root = std::make_unique<HMatrix<T, htool::underlying_type<T>>>(hmatrix_tree_builder_A.build(policy_root, *test_case.operator_in_user_numbering_A, *test_case.root_cluster_A_output, *test_case.root_cluster_A_input));
        lu_factorization(policy_root, *A_root);
    }
    is_error    = is_error || policy_root.hmatrix_task_dependencies.L0.size() != 1 || policy_root.hmatrix_task_dependencies.L0[0] != A_root.get();
    matrix_test = B_dense;
    lu_solve(trans, *A_root, matrix_test);
    error    = normFrob(X_dense - matrix_test) / normFrob(X_dense);
    is_error = is_error || !(error < epsilon * margin);
    cout << "> Errors on hmatrix lu solve with L0 reduced to root: " << error << '\n';
    cout << "> is_error: " << is_error << "\n";

    // An L0 of a sub-block is not an L0 of the whole HMatrix, but an L0 assigned directly is kept if it is one
    omp_task_policy<T, double> policy_custom;
    HMatrix<T, htool::underlying_type<T>> A_custom = hmatrix_tree_builder_A.build(*test_case.operator_in_user_numbering_A, *test_case.root_cluster_A_output, *test_case.root_cluster_A_input);
    auto &dependencies_custom                      = policy_custom.hmatrix_task_dependencies;
    dependencies_custom.set_L0(*A_custom.get_children()[0]);
    is_error               = is_error || dependencies_custom.is_L0_of(A_custom);
    dependencies_custom.L0 = find_l0(A_custom, 16, dependencies_custom.cost_function);
    auto L0_custom         = dependencies_custom.L0;
    is_error               = is_error || !dependencies_custom.is_L0_of(A_custom);
    lu_factorization(policy_custom, A_custom);
    is_error    = is_error || dependencies_custom.L0 != L0_custom;
    matrix_test = B_dense;
    lu_solve(trans, A_custom, matrix_test);
    error    = normFrob(X_dense - matrix_test) / normFrob(X_dense);
    is_error = is_error || !(error < epsilon * margin);
    cout << "> Errors on hmatrix lu solve with an L0 assigned directly: " << error << '\n';

    // L0 recomputed by a factorization while the tasks of the build are pending: they must be completed first
    omp_task_policy<T, double> policy_recomputed;
    std::unique_ptr<HMatrix<T, htool::underlying_type<T>>> A_recomputed;
#if defined(_OPENMP)
#    pragma omp parallel
#    pragma omp single
#endif
    {
        A_recomputed = std::make_unique<HMatrix<T, htool::underlying_type<T>>>(hmatrix_tree_builder_A.build(policy_recomputed, *test_case.operator_in_user_numbering_A, *test_case.root_cluster_A_output, *test_case.root_cluster_A_input));
        policy_recomputed.hmatrix_task_dependencies.L0.resize(1);            // no longer a cut of A_recomputed's block tree, so that it is recomputed,
        policy_recomputed.hmatrix_task_dependencies.max_number_of_nodes = 8; // with other nodes than the ones the build tasks depend on
        lu_factorization(policy_recomputed, *A_recomputed);
    }
    matrix_test = B_dense;
    lu_solve(trans, *A_recomputed, matrix_test);
    error    = normFrob(X_dense - matrix_test) / normFrob(X_dense);
    is_error = is_error || !(error < epsilon * margin);
    cout << "> Errors on hmatrix lu solve with L0 recomputed after the build: " << error << '\n';
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
        A = std::make_unique<HMatrix<T, htool::underlying_type<T>>>(hmatrix_tree_builder_A.build(policy, *test_case.operator_in_user_numbering_A, *test_case.root_cluster_A_output, *test_case.root_cluster_A_input));

        // Factorization
        cholesky_factorization(policy, UPLO, *A);
    }

    // Cholesky factorization
    matrix_test = B_dense;
    cholesky_solve(UPLO, *A, matrix_test);
    error    = normFrob(X_dense - matrix_test) / normFrob(X_dense);
    is_error = is_error || !(error < epsilon * margin);
    cout << "> Errors on hmatrix cholesky solve: " << error << '\n';
    cout << "> is_error: " << is_error << "\n";

    return is_error;
}

template <typename T, typename GeneratorTestType>
bool test_task_based_hmatrix_full_ldlt(char UPLO, char symmetry, int n1, int n2, htool::underlying_type<T> epsilon, htool::underlying_type<T> margin) {
    bool is_error = false;
    double eta    = 100;
    htool::underlying_type<T> error;

    // Setup test case
    htool::TestCaseSolve<T, GeneratorTestType> test_case('L', 'N', n1, n2, 1, -1);

    // Policy
    omp_task_policy<T, double> policy;

    // HMatrix
    HMatrixTreeBuilder<T, htool::underlying_type<T>> hmatrix_tree_builder_A(epsilon, eta, symmetry, UPLO);

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
    if (symmetry == 'H') {
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
        A = std::make_unique<HMatrix<T, htool::underlying_type<T>>>(hmatrix_tree_builder_A.build(policy, *test_case.operator_in_user_numbering_A, *test_case.root_cluster_A_output, *test_case.root_cluster_A_input));

        // Factorization
        ldlt_factorization(policy, symmetry, UPLO, *A);
    }

    // LDLt solve
    matrix_test = B_dense;
    ldlt_solve(symmetry, UPLO, *A, matrix_test);
    error    = normFrob(X_dense - matrix_test) / normFrob(X_dense);
    is_error = is_error || !(error < epsilon * margin);
    cout << "> Errors on hmatrix " << (symmetry == 'H' ? "hermitian" : "symmetric") << " ldlt solve: " << error << '\n';
    cout << "> is_error: " << is_error << "\n";

    return is_error;
}

// Geometric cluster tree on unevenly distributed points: some clusters have a leaf child next to a non-leaf one, so that some blocks
// are hierarchical with a dense diagonal block as pivot, which the triangular solves must not densify while tasks are created.
template <typename T>
bool test_task_based_hmatrix_factorizations_unbalanced_cluster_tree() {
    bool is_error = false;
    const int n   = 1500;
    std::mt19937 rng(0);
    std::uniform_real_distribution<double> uniform(0, 1);
    std::vector<double> points(3 * n);
    for (int i = 0; i < n; i++) {
        for (int d = 0; d < 3; d++) {
            points[3 * i + d] = (i < n - 40 ? uniform(rng) : 20 + 0.1 * uniform(rng)) + (d == 0 && i % 7 == 0 ? 3 : 0);
        }
    }
    ClusterTreeBuilder<double> cluster_builder;
    cluster_builder.set_maximal_leaf_size(20);
    cluster_builder.set_partitioning_strategy(std::make_shared<Partitioning<double, ComputeLargestExtent<double>, GeometricSplitting<double>>>());
    Cluster<double> cluster = cluster_builder.create_cluster_tree(n, 3, points.data(), 2, 1);

    bool is_unbalanced = false;
    preorder_tree_traversal(cluster, [&](const Cluster<double> &current_cluster) {
        const auto &children = current_cluster.get_children();
        is_unbalanced        = is_unbalanced || (std::any_of(children.begin(), children.end(), [](const auto &child) { return child->is_leaf(); }) && std::any_of(children.begin(), children.end(), [](const auto &child) { return !child->is_leaf(); }));
    });
    is_error = is_error || !is_unbalanced;

    Matrix<T> A_dense(n, n), X_dense(n, 5), B_dense(n, 5), matrix_test;
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            double r      = std::sqrt(std::pow(points[3 * i] - points[3 * j], 2) + std::pow(points[3 * i + 1] - points[3 * j + 1], 2) + std::pow(points[3 * i + 2] - points[3 * j + 2], 2));
            A_dense(i, j) = std::exp(-r) + (i == j ? 1 : 0);
        }
    }
    GeneratorInUserNumberingFromMatrix<T> generator(A_dense);
    generate_random_matrix(X_dense);
    add_matrix_matrix_product('N', 'N', T(1), A_dense, X_dense, T(0), B_dense);

    auto check = [&](const std::string &name, char symmetry, char UPLO, auto &&factorize_and_solve) {
        HMatrixTreeBuilder<T, htool::underlying_type<T>> hmatrix_tree_builder(1e-8, 10, symmetry, UPLO);
        HMatrix<T, htool::underlying_type<T>> hmatrix = hmatrix_tree_builder.build(generator, cluster, cluster);
        matrix_test                                   = B_dense;
        factorize_and_solve(hmatrix, matrix_test);
        htool::underlying_type<T> error = normFrob(X_dense - matrix_test) / normFrob(X_dense);
        is_error                        = is_error || !(error < 1e-6);
        cout << "> Errors on hmatrix " << name << " solve with an unbalanced cluster tree: " << error << '\n';
    };
    char symmetry = is_complex<T>() ? 'H' : 'S';
    check("lu", 'N', 'N', [](auto &hmatrix, auto &X) { omp_task_policy<T, double> policy; lu_factorization(policy, hmatrix); lu_solve('N', hmatrix, X); });
    for (char UPLO : {'L', 'U'}) {
        check("cholesky", symmetry, UPLO, [UPLO](auto &hmatrix, auto &X) { omp_task_policy<T, double> policy; cholesky_factorization(policy, UPLO, hmatrix); cholesky_solve(UPLO, hmatrix, X); });
        // the matrix is real, hence also complex symmetric
        for (char ldlt_symmetry : is_complex<T>() ? std::vector<char>{'S', 'H'} : std::vector<char>{'S'}) {
            check(std::string("ldlt (") + ldlt_symmetry + ")", ldlt_symmetry, UPLO, [UPLO, ldlt_symmetry](auto &hmatrix, auto &X) { omp_task_policy<T, double> policy; ldlt_factorization(policy, ldlt_symmetry, UPLO, hmatrix); ldlt_solve(ldlt_symmetry, UPLO, hmatrix, X); });
        }
    }
    cout << "> is_error: " << is_error << "\n";
    return is_error;
}
