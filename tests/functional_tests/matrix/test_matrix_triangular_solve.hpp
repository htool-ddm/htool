#include <htool/matrix/linalg/add_matrix_matrix_product.hpp> // for add_mat...
#include <htool/matrix/linalg/factorization.hpp>             // for triangu...
#include <htool/matrix/matrix.hpp>                           // for Matrix
#include <htool/misc/misc.hpp>                               // for underly...
#include <htool/testing/generator_input.hpp>                 // for generat...
#include <iostream>                                          // for operator<<
#include <vector>                                            // for vector
using namespace std;
using namespace htool;

template <typename T>
bool test_matrix_triangular_solve(int n, int nrhs, char side, char transa) {

    bool is_error = false;

    // Generate random matrix
    htool::underlying_type<T> error;
    T alpha;
    Matrix<T> result(n, nrhs), B(n, nrhs);
    if (side == 'R') {
        result.resize(nrhs, n);
        B.resize(nrhs, n);
    }
    generate_random_array(result.data(), result.nb_rows() * result.nb_cols());
    generate_random_scalar(alpha);

    Matrix<T> A(n, n), test_factorization, test_solve;
    generate_random_array(A.data(), A.nb_rows() * A.nb_cols());
    for (int i = 0; i < n; i++) {
        T sum = 0;
        for (int j = 0; j < n; j++) {
            sum += std::abs(A(i, j));
        }
        A(i, i) = sum;
    }

    // Triangular setup
    Matrix<T> LA(A), UA(A), LB(B.nb_rows(), B.nb_cols()), permuted_LB(B.nb_rows(), B.nb_cols()), UB(B.nb_rows(), B.nb_cols());
    for (int i = 0; i < A.nb_rows(); i++) {
        for (int j = 0; j < A.nb_cols(); j++) {
            if (i > j) {
                UA(i, j) = 0;
            }
            if (i < j) {
                LA(i, j) = 0;
            }
        }
    }

    // Permutation
    std::vector<int> ipiv(A.nb_rows()), inverse_permutation(A.nb_rows());
    for (int i = 0; i < A.nb_rows(); i++) {
        generate_random_scalar(ipiv[i], 0, A.nb_rows() - i - 1);
    }
    int count = 1;
    for (int i = 0; i < A.nb_rows(); i++) {
        ipiv[i] += count;
        count += 1;
    }

    if (side == 'L') {
        Matrix<T> temp_result(result);
        if (transa != 'N') {
            for (int i = 0; i < LB.nb_rows(); i++) {
                for (int j = 0; j < LB.nb_cols(); j++) {
                    std::swap(temp_result(ipiv[i] - 1, j), temp_result(i, j));
                }
            }
        }

        add_matrix_matrix_product(transa, 'N', T(1) / alpha, UA, result, T(0), UB);
        add_matrix_matrix_product(transa, 'N', T(1) / alpha, LA, result, T(0), LB);

        if (transa == 'N') {
            permuted_LB = LB;
            for (int i = LB.nb_rows() - 1; i >= 0; i--) {
                for (int j = 0; j < LB.nb_cols(); j++) {
                    std::swap(permuted_LB(i, j), permuted_LB(ipiv[i] - 1, j));
                }
            }
        } else {
            add_matrix_matrix_product(transa, 'N', T(1) / alpha, LA, temp_result, T(0), permuted_LB);
        }
    } else if (side == 'R') {
        Matrix<T> temp_result(result);
        if (transa == 'N') {
            for (int i = 0; i < LB.nb_rows(); i++) {
                for (int j = 0; j < LB.nb_cols(); j++) {
                    std::swap(temp_result(i, ipiv[j] - 1), temp_result(i, j));
                }
            }
        }

        add_matrix_matrix_product('N', transa, T(1) / alpha, result, UA, T(0), UB);
        add_matrix_matrix_product('N', transa, T(1) / alpha, result, LA, T(0), LB);

        if (transa == 'N') {
            add_matrix_matrix_product('N', transa, T(1) / alpha, temp_result, LA, T(0), permuted_LB);
        } else {
            permuted_LB = LB;
            for (int i = LB.nb_rows() - 1; i >= 0; i--) {
                for (int j = LB.nb_cols() - 1; j >= 0; j--) {
                    std::swap(permuted_LB(i, j), permuted_LB(i, ipiv[j] - 1));
                }
            }
        }
    }

    // triangular_matrix_matrix_solve
    test_factorization = LA;
    test_solve         = LB;
    triangular_matrix_matrix_solve(side, 'L', transa, 'N', alpha, test_factorization, test_solve);
    error    = normFrob(result - test_solve) / normFrob(result);
    is_error = is_error || !(error < 1e-9);
    cout << "> Errors on lower triangular matrix matrix solve: " << error << endl;

    test_factorization              = LA;
    test_factorization.get_pivots() = ipiv;
    test_solve                      = permuted_LB;
    triangular_matrix_matrix_solve(side, 'L', transa, 'N', alpha, test_factorization, test_solve);
    error    = normFrob(result - test_solve) / normFrob(result);
    is_error = is_error || !(error < 1e-9);
    cout << "> Errors on lower triangular matrix matrix solve with permutation: " << error << endl;

    test_factorization = UA;
    test_solve         = UB;
    triangular_matrix_matrix_solve(side, 'U', transa, 'N', alpha, test_factorization, test_solve);
    error    = normFrob(result - test_solve) / normFrob(result);
    is_error = is_error || !(error < 1e-9);
    cout << "> Errors on upper triangular matrix matrix solve: " << error << endl;

    return is_error;
}

template <typename T>
bool test_symmetric_matrix_triangular_solve(int n, int nrhs, char side) {

    bool is_error = false;

    // Generate random matrix
    htool::underlying_type<T> error;
    T alpha;
    Matrix<T> result(n, nrhs), B(n, nrhs);
    if (side == 'R') {
        result.resize(nrhs, n);
        B.resize(nrhs, n);
    }
    generate_random_array(result.data(), result.nb_rows() * result.nb_cols());
    generate_random_scalar(alpha);

    Matrix<T> A(n, n), LDLtA, UDUtA, LLtA, UtUA, test_factorization, test_solve;
    Matrix<T> random_matrix(n, n);
    generate_random_array(random_matrix.data(), random_matrix.nb_rows() * random_matrix.nb_cols());

    add_matrix_matrix_product('T', 'N', T(1), random_matrix, random_matrix, T(1), A);
    T eps = *std::max_element(random_matrix.data(), random_matrix.data() + random_matrix.nb_cols() * random_matrix.nb_rows(), [](const T &lhs, const T &rhs) { return std::abs(lhs) < std::abs(rhs); });
    for (int i = 0; i < n; i++) {
        A(i, i) += std::abs(eps);
    }
    LDLtA = A;
    UDUtA = A;
    LLtA  = A;
    UtUA  = A;

    if (side == 'L') {
        add_matrix_matrix_product('N', 'N', T(1), A, result, T(0), B);
    } else {
        add_matrix_matrix_product('N', 'N', T(1), result, A, T(0), B);
    }

    symmetric_ldlt_factorization('L', LDLtA);
    symmetric_ldlt_factorization('U', UDUtA);
    cholesky_factorization('L', LLtA);
    cholesky_factorization('U', UtUA);

    // Solve
    test_factorization = LLtA;
    test_solve         = B;
    triangular_matrix_matrix_solve(side, 'L', side == 'L' ? 'N' : 'T', 'N', 1., test_factorization, test_solve);
    triangular_matrix_matrix_solve(side, 'L', side == 'L' ? 'T' : 'N', 'N', 1., test_factorization, test_solve);
    error    = normFrob(result - test_solve) / normFrob(result);
    is_error = is_error || !(error < 1e-9);
    cout << "> Errors on lower cholesky matrix matrix solve: " << error << endl;

    test_factorization = UtUA;
    test_solve         = B;
    triangular_matrix_matrix_solve(side, 'U', side == 'L' ? 'T' : 'N', 'N', 1., test_factorization, test_solve);
    triangular_matrix_matrix_solve(side, 'U', side == 'L' ? 'N' : 'T', 'N', 1., test_factorization, test_solve);
    error    = normFrob(result - test_solve) / normFrob(result);
    is_error = is_error || !(error < 1e-9);
    cout << "> Errors on upper cholesky matrix matrix solve: " << error << endl;

    test_factorization = LDLtA;
    test_solve         = B;
    triangular_ldlt_matrix_matrix_solve(side, 'L', side == 'L' ? 'N' : 'T', 'N', 1., test_factorization, test_solve);
    triangular_ldlt_matrix_matrix_solve(side, 'L', side == 'L' ? 'T' : 'N', 'N', 1., test_factorization, test_solve);
    // test_solve.print(std::cout, ",");
    // (result - test_solve).print(std::cout, ",");
    std::cout << normFrob(result - test_solve) << " " << normFrob(result) << " " << normFrob(test_solve) << "\n";
    error    = normFrob(result - test_solve) / normFrob(result);
    is_error = is_error || !(error < 1e-9);
    cout << "> Errors on lower triangular ldlt matrix matrix solve: " << error << endl;

    test_factorization = UDUtA;
    test_solve         = B;
    triangular_ldlt_matrix_matrix_solve(side, 'U', 'N', 'N', 1., test_factorization, test_solve);
    triangular_ldlt_matrix_matrix_solve(side, 'U', 'T', 'N', 1., test_factorization, test_solve);
    error    = normFrob(result - test_solve) / normFrob(result);
    is_error = is_error || !(error < 1e-9);
    cout << "> Errors on upper triangular ldlt matrix matrix solve: " << error << endl;

    return is_error;
}
