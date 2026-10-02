#include <algorithm>                         // for max_ele...
#include <htool/matrix/linalg.hpp>           // for cholesk...
#include <htool/matrix/matrix.hpp>           // for Matrix
#include <htool/matrix/utils.hpp>            // for normFrob
#include <htool/misc/misc.hpp>               // for underly...
#include <htool/testing/generator_input.hpp> // for generat...
#include <iostream>                          // for basic_o...

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
    cout << "> Errors on matrix lu solve: " << error << '\n';

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
    // Diagonal-dominance boost (Gershgorin: guarantees nonsingularity and a bounded condition
    // number, unlike a flat constant that doesn't scale with n). Not strictly required here -
    // random^{op}*random is already PSD for any T when op is the conjugate transpose - but kept
    // for consistency and to keep the condition number away from n's growth.
    for (int i = 0; i < n; i++) {
        htool::underlying_type<T> row_sum = 0;
        for (int j = 0; j < n; j++) {
            if (j != i) {
                row_sum += std::abs(A(i, j));
            }
        }
        A(i, i) += row_sum;
    }

    add_matrix_matrix_product(trans, 'N', T(1), A, result, T(1), B);

    // LU factorization
    test_factorization = A;
    test_solve         = B;
    cholesky_factorization(UPLO, test_factorization);
    cholesky_solve(UPLO, test_factorization, test_solve);
    error    = normFrob(result - test_solve) / normFrob(result);
    is_error = is_error || !(error < 1e-9);
    cout << "> Errors on matrix cholesky solve: " << error << '\n';

    return is_error;
}

// well_conditioned selects between a diagonally-dominant A (Gershgorin: guarantees
// nonsingularity and a bounded condition number - reliable, checked at a tight tolerance) and
// the original weakly-boosted A (random^T*random has no positive-definiteness guarantee for
// complex T, so this one genuinely forces 2x2 Bunch-Kaufman pivots, at a looser tolerance since
// the matrix itself is ill-conditioned).
template <typename T>
bool test_matrix_symmetric_ldlt(char trans, int n, int nrhs, char UPLO, bool well_conditioned) {

    bool is_error = false;

    // Generate random matrix
    htool::underlying_type<T> error;
    Matrix<T> result(n, nrhs), B(n, nrhs);
    generate_random_array(result.data(), result.nb_rows() * result.nb_cols());

    Matrix<T> A(n, n), test_factorization, test_solve;
    Matrix<T> random_matrix(n, n);
    generate_random_array(random_matrix.data(), random_matrix.nb_rows() * random_matrix.nb_cols());
    char op = 'T';
    add_matrix_matrix_product(op, 'N', T(1), random_matrix, random_matrix, T(1), A);
    htool::underlying_type<T> tol;
    if (well_conditioned) {
        for (int i = 0; i < n; i++) {
            htool::underlying_type<T> row_sum = 0;
            for (int j = 0; j < n; j++) {
                if (j != i) {
                    row_sum += std::abs(A(i, j));
                }
            }
            A(i, i) += row_sum;
        }
        tol = 1e-9;
    } else {
        T eps = *std::max_element(random_matrix.data(), random_matrix.data() + random_matrix.nb_cols() * random_matrix.nb_rows(), [](const T &lhs, const T &rhs) { return std::abs(lhs) < std::abs(rhs); });
        for (int i = 0; i < n; i++) {
            A(i, i) += std::abs(eps);
        }
        tol = 1e-6;
    }

    add_matrix_matrix_product(trans, 'N', T(1), A, result, T(1), B);

    // LU factorization
    test_factorization = A;
    test_solve         = B;
    ldlt_factorization('S', UPLO, test_factorization);

    if (!well_conditioned && htool::is_complex<T>()) {
        bool has_2x2 = false;
        for (auto v : test_factorization.get_pivots()) {
            has_2x2 = has_2x2 || v < 0;
        }
        if (!has_2x2) {
            is_error = true;                                                                                                                  // LCOV_EXCL_LINE
            cout << "> WARNING: ill-conditioned symmetric ldlt never triggered a 2x2 Bunch-Kaufman pivot - lost 2x2 branch coverage" << '\n'; // LCOV_EXCL_LINE
        }
    }

    ldlt_solve('S', UPLO, test_factorization, test_solve);
    error    = normFrob(result - test_solve) / normFrob(result);
    is_error = is_error || !(error < tol);
    cout << "> Errors on matrix symmetric LDLt solve: " << error << '\n';

    return is_error;
}

template <typename T>
bool test_matrix_hermitian_ldlt(char trans, int n, int nrhs, char UPLO) {

    bool is_error = false;

    // Generate random matrix
    htool::underlying_type<T> error;
    Matrix<T> result(n, nrhs), B(n, nrhs);
    generate_random_array(result.data(), result.nb_rows() * result.nb_cols());

    Matrix<T> A(n, n), test_factorization, test_solve;
    Matrix<T> random_matrix(n, n);
    generate_random_array(random_matrix.data(), random_matrix.nb_rows() * random_matrix.nb_cols());
    char op = 'C';
    add_matrix_matrix_product(op, 'N', T(1), random_matrix, random_matrix, T(1), A);
    // Diagonal-dominance boost (Gershgorin). Not strictly required - random^H*random is already
    // PSD for any T - but kept for consistency; empirically hetrf never picks a 2x2 pivot for
    // this construction either way (Hermitian PSD, unlike the plain-transpose symmetric case).
    for (int i = 0; i < n; i++) {
        htool::underlying_type<T> row_sum = 0;
        for (int j = 0; j < n; j++) {
            if (j != i) {
                row_sum += std::abs(A(i, j));
            }
        }
        A(i, i) += row_sum;
    }

    add_matrix_matrix_product(trans, 'N', T(1), A, result, T(1), B);

    // LU factorization
    test_factorization = A;
    test_solve         = B;
    ldlt_factorization('H', UPLO, test_factorization);
    ldlt_solve('H', UPLO, test_factorization, test_solve);
    error    = normFrob(result - test_solve) / normFrob(result);
    is_error = is_error || !(error < 1e-9);
    cout << "> Errors on matrix hermitian LDLt solve: " << error << '\n';

    return is_error;
}
