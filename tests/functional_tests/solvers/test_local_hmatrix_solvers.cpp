#include <complex>                                           // for complex
#include <htool/hmatrix/tree_builder/tree_builder.hpp>       // for HMatrixTreeBuilder
#include <htool/matrix/linalg/add_matrix_matrix_product.hpp> // for add_matrix_matrix_product
#include <htool/matrix/utils/math.hpp>                       // for normFrob
#include <htool/solvers/local_solvers/local_hmatrix_plus_overlap_solvers.hpp>
#include <htool/solvers/local_solvers/local_hmatrix_solvers.hpp>
#include <htool/testing/generate_test_case.hpp> // for TestCaseSolve
#include <htool/testing/generator_input.hpp>    // for generate_random_matrix
#include <htool/testing/generator_test.hpp>     // for GeneratorInUserNumberingFromMatrix
#include <iostream>                             // for cout
#include <mpi.h>                                // for MPI_Init

using namespace htool;

// Solves K x = b with the local solvers, K = [A, B; op(B), D] (op = ^T for symmetry 'S', ^H for 'H').
// With positive_definite, A and D are diagonally dominant with a positive diagonal. Otherwise (and
// always for a complex symmetric matrix, which cannot be positive definite), consecutive rows of
// each cluster leaf are paired into dominant [[0,d],[op(d),0]] blocks, so that A is indefinite
// and LDLt needs 2x2 pivots.
template <typename T, typename GeneratorTestType>
bool test_local_hmatrix_solvers(char symmetry, char UPLO, bool positive_definite) {
    bool is_error = false;
    int n_overlap = 40, mu = 3;
    double tol  = 1e-8;
    bool paired = !positive_definite || (symmetry == 'S' && is_complex<T>());
    auto op     = [symmetry](T value) { return symmetry == 'H' ? conj_if_complex(value) : value; };

    TestCaseSolve<T, GeneratorTestType> test_case('L', 'N', 300, 10, 1, -1);
    const Cluster<underlying_type<T>> &cluster = *test_case.root_cluster_A_output;
    const std::vector<int> &permutation        = cluster.get_permutation();
    int n                                      = cluster.get_size();

    Matrix<T> A_user(n, n);
    std::vector<int> identity(n);
    std::iota(identity.begin(), identity.end(), 0);
    test_case.operator_in_user_numbering_A->copy_submatrix(n, n, identity.data(), identity.data(), A_user.data());
    if (paired) {
        T omega(0.6);
        if constexpr (is_complex<T>()) {
            omega += T(0, 0.8);
        }
        preorder_tree_traversal(cluster, [&](const Cluster<underlying_type<T>> &node) {
            if (node.is_leaf()) {
                for (int k = node.get_offset(); k + 1 < node.get_offset() + node.get_size(); k += 2) {
                    int u = permutation[k], v = permutation[k + 1];
                    T d          = A_user(u, u) * omega;
                    A_user(u, v) = d;
                    A_user(v, u) = op(d);
                    A_user(u, u) = 0;
                    A_user(v, v) = 0;
                }
            }
        });
    }
    GeneratorInUserNumberingFromMatrix<T> generator(A_user);

    // K in cluster numbering
    Matrix<T> K(n + n_overlap, n + n_overlap), B(n, n_overlap), D(n_overlap, n_overlap);
    generate_random_matrix(B);
    generate_random_matrix(D);
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            K(i, j) = A_user(permutation[i], permutation[j]);
        }
    }
    for (int j = 0; j < n_overlap; j++) {
        for (int i = 0; i < n; i++) {
            K(i, n + j) = B(i, j);
            K(n + j, i) = op(B(i, j));
        }
        for (int i = 0; i < n_overlap; i++) {
            K(n + i, n + j) = i == j ? T(1e5) : (i < j ? D(i, j) : op(D(j, i)));
        }
    }
    Matrix<T> b(n + n_overlap, mu);
    generate_random_matrix(b);
    HPDDM::MatrixCSR<T> *no_csr = nullptr;
    HMatrixTreeBuilder<T, underlying_type<T>> hmatrix_builder(1e-10, 10, symmetry, UPLO);

    // With overlap
    {
        HMatrix<T, underlying_type<T>> local_hmatrix = hmatrix_builder.build(generator, cluster, cluster);
        Matrix<T> B_overlap(n, n_overlap), C_overlap(n_overlap, n), D_overlap(n_overlap, n_overlap);
        for (int j = 0; j < n_overlap; j++) {
            for (int i = 0; i < n; i++) {
                B_overlap(i, j) = K(i, n + j);
                C_overlap(j, i) = K(n + j, i);
            }
            for (int i = 0; i < n_overlap; i++) {
                D_overlap(i, j) = K(n + i, n + j);
            }
        }
        LocalHMatrixPlusOverlapSolver<T, underlying_type<T>> solver(local_hmatrix, B_overlap, C_overlap, D_overlap, positive_definite);
        solver.numfact(no_csr);
        Matrix<T> x(b), residual(b);
        solver.solve(x.data(), mu);
        add_matrix_matrix_product('N', 'N', T(-1), K, x, T(1), residual);
        underlying_type<T> error = normFrob(residual) / normFrob(b);
        is_error                 = is_error || !(error < tol);
        std::cout << "> Residual of local hmatrix plus overlap solver (" << symmetry << ", " << UPLO << ", positive definite: " << positive_definite << "): " << error << '\n';
    }

    // Without overlap
    {
        HMatrix<T, underlying_type<T>> local_hmatrix = hmatrix_builder.build(generator, cluster, cluster);
        LocalHMatrixSolver<T, underlying_type<T>> solver(local_hmatrix, false, positive_definite);
        solver.numfact(no_csr);
        Matrix<T> A(n, n), b_local(n, mu);
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < n; i++) {
                A(i, j) = K(i, j);
            }
        }
        for (int j = 0; j < mu; j++) {
            for (int i = 0; i < n; i++) {
                b_local(i, j) = b(i, j);
            }
        }
        Matrix<T> x(b_local), residual(b_local);
        solver.solve(x.data(), mu);
        add_matrix_matrix_product('N', 'N', T(-1), A, x, T(1), residual);
        underlying_type<T> error = normFrob(residual) / normFrob(b_local);
        is_error                 = is_error || !(error < tol);
        std::cout << "> Residual of local hmatrix solver (" << symmetry << ", " << UPLO << ", positive definite: " << positive_definite << "): " << error << '\n';
    }
    return is_error;
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    bool is_error = false;
    for (char UPLO : {'L', 'U'}) {
        for (bool positive_definite : {false, true}) {
            is_error = is_error || test_local_hmatrix_solvers<double, GeneratorTestDoubleSymmetric>('S', UPLO, positive_definite);
            is_error = is_error || test_local_hmatrix_solvers<std::complex<double>, GeneratorTestComplexSymmetric>('S', UPLO, positive_definite);
            is_error = is_error || test_local_hmatrix_solvers<std::complex<double>, GeneratorTestComplexHermitian>('H', UPLO, positive_definite);
        }
    }
    MPI_Finalize();
    return is_error ? 1 : 0;
}
