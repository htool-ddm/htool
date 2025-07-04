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
bool test_matrix_triangular_solve(int n, int nrhs, char side, char transa, char diag) {

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
    if (diag == 'U') {
        T max = 0;
        for (int i = 0; i < n; i++) {
            max = std::max(std::abs(max), std::abs(A(i, i)));
        }
        scale(1. / max, A);
    }
    // Triangular setup
    Matrix<T> LA(A), UA(A), LB(B.nb_rows(), B.nb_cols()), permuted_LB(B.nb_rows(), B.nb_cols()), UB(B.nb_rows(), B.nb_cols());
    for (int i = 0; i < A.nb_rows(); i++) {
        if (diag == 'U') {
            UA(i, i) = 1;
            LA(i, i) = 1;
        }
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
    triangular_matrix_matrix_solve(side, 'L', transa, diag, alpha, test_factorization, test_solve);
    // test_solve.print(std::cout, ",");
    error    = normFrob(result - test_solve) / normFrob(result);
    is_error = is_error || !(error < 1e-9);
    cout << "> Errors on lower triangular matrix matrix solve: " << error << endl;

    test_factorization              = LA;
    test_factorization.get_pivots() = ipiv;
    // std::cout << ipiv.size() << "\n";
    // for (auto elt : ipiv) {
    //     std::cout << elt << " ";
    // }
    // std::cout << "\n";
    test_solve = permuted_LB;
    triangular_matrix_matrix_solve(side, 'L', transa, diag, alpha, test_factorization, test_solve);
    error    = normFrob(result - test_solve) / normFrob(result);
    is_error = is_error || !(error < 1e-9);
    cout << "> Errors on lower triangular matrix matrix solve with permutation: " << error << endl;

    test_factorization = UA;
    test_solve         = UB;
    triangular_matrix_matrix_solve(side, 'U', transa, diag, alpha, test_factorization, test_solve);
    error    = normFrob(result - test_solve) / normFrob(result);
    is_error = is_error || !(error < 1e-9);
    cout << "> Errors on upper triangular matrix matrix solve: " << error << endl;

    return is_error;
}
