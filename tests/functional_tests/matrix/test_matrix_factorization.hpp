#include <algorithm>                                         // for max_ele...
#include <htool/matrix/linalg/add_matrix_matrix_product.hpp> // for add_mat...
#include <htool/matrix/linalg/factorization.hpp>             // for cholesk...
#include <htool/matrix/matrix.hpp>                           // for Matrix
#include <htool/misc/misc.hpp>                               // for underly...
#include <htool/testing/generator_input.hpp>                 // for generat...
#include <iostream>                                          // for basic_o...

using namespace std;
using namespace htool;

template <typename T>
bool test_matrix_lu(char trans, int n, int nrhs) {

    bool is_error = false;

    // Generate random matrix
    htool::underlying_type<T> error;
    Matrix<T> result(n, nrhs), B(n, nrhs);
    generate_random_array(result.data(), result.nb_rows() * result.nb_cols());

    Matrix<T> A(n, n), test_factorization, test_solve;
    generate_random_array(A.data(), A.nb_rows() * A.nb_cols());
    for (int i = 0; i < n; i++) {
        T sum = 0;
        for (int j = 0; j < n; j++) {
            sum += std::abs(A(i, j));
        }
        A(i, i) = sum;
    }

    add_matrix_matrix_product(trans, 'N', T(1), A, result, T(1), B);

    // LU factorization
    test_factorization = A;
    test_solve         = B;
    lu_factorization(test_factorization);
    lu_solve(trans, test_factorization, test_solve);
    error    = normFrob(result - test_solve) / normFrob(result);
    is_error = is_error || !(error < 1e-9);
    cout << "> Errors on matrix lu solve: " << error << endl;

    return is_error;
}

template <typename T>
bool test_matrix_cholesky(char trans, int n, int nrhs, char symmetry, char UPLO) {

    bool is_error = false;

    // Generate random matrix
    htool::underlying_type<T> error;
    Matrix<T> result(n, nrhs), B(n, nrhs);
    generate_random_array(result.data(), result.nb_rows() * result.nb_cols());

    Matrix<T> A(n, n), test_factorization, test_solve;
    Matrix<T> random_matrix(n, n);
    generate_random_array(random_matrix.data(), random_matrix.nb_rows() * random_matrix.nb_cols());
    char op = symmetry == 'S' ? 'T' : 'C';
    add_matrix_matrix_product(op, 'N', T(1), random_matrix, random_matrix, T(1), A);
    T eps = *std::max_element(random_matrix.data(), random_matrix.data() + random_matrix.nb_cols() * random_matrix.nb_rows(), [](const T &lhs, const T &rhs) { return std::abs(lhs) < std::abs(rhs); });
    for (int i = 0; i < n; i++) {
        A(i, i) += std::abs(eps);
    }

    add_matrix_matrix_product(trans, 'N', T(1), A, result, T(1), B);

    // LU factorization
    test_factorization = A;
    test_solve         = B;
    cholesky_factorization(UPLO, test_factorization);
    cholesky_solve(UPLO, test_factorization, test_solve);
    error    = normFrob(result - test_solve) / normFrob(result);
    is_error = is_error || !(error < 1e-9);
    cout << "> Errors on matrix cholesky solve: " << error << endl;

    return is_error;
}
