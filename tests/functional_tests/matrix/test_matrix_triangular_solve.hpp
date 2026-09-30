
#include <htool/matrix/linalg.hpp>           // for triangu...
#include <htool/matrix/matrix.hpp>           // for Matrix
#include <htool/matrix/utils.hpp>            // for triangu...
#include <htool/misc/misc.hpp>               // for underly...
#include <htool/testing/generator_input.hpp> // for generat...
#include <iostream>                          // for operator<<
#include <vector>                            // for vector
using namespace std;
using namespace htool;

template <typename T>
bool test_matrix_triangular_solve(int n, int nrhs, char side, char transa, char diag) {

    bool is_error = false;

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

    test_factorization = LA;
    test_solve         = LB;
    triangular_matrix_matrix_solve(side, 'L', transa, diag, alpha, test_factorization, test_solve);
    error    = normFrob(result - test_solve) / normFrob(result);
    is_error = is_error || !(error < 1e-9);
    cout << "> Errors on lower triangular matrix matrix solve: " << error << '\n';

    test_factorization              = LA;
    test_factorization.get_pivots() = ipiv;
    test_solve                      = permuted_LB;
    triangular_matrix_matrix_solve(side, 'L', transa, diag, alpha, test_factorization, test_solve);
    error    = normFrob(result - test_solve) / normFrob(result);
    is_error = is_error || !(error < 1e-9);
    cout << "> Errors on lower triangular matrix matrix solve with permutation: " << error << '\n';

    test_factorization = UA;
    test_solve         = UB;
    triangular_matrix_matrix_solve(side, 'U', transa, diag, alpha, test_factorization, test_solve);
    error    = normFrob(result - test_solve) / normFrob(result);
    is_error = is_error || !(error < 1e-9);
    cout << "> Errors on upper triangular matrix matrix solve: " << error << '\n';

    return is_error;
}

template <typename T>
bool test_cholesky_matrix_triangular_solve(int n, int nrhs, char side) {

    bool is_error = false;

    htool::underlying_type<T> error;
    T alpha;
    Matrix<T> result(n, nrhs), B(n, nrhs);
    if (side == 'R') {
        result.resize(nrhs, n);
        B.resize(nrhs, n);
    }
    generate_random_array(result.data(), result.nb_rows() * result.nb_cols());
    generate_random_scalar(alpha);

    Matrix<T> A(n, n), LLtA, UtUA, test_factorization, test_solve;
    Matrix<T> random_matrix(n, n);
    generate_random_array(random_matrix.data(), random_matrix.nb_rows() * random_matrix.nb_cols());

    // 'C' degrades to plain transpose for real T: random^H*random is Hermitian PSD for any T.
    add_matrix_matrix_product('C', 'N', T(1), random_matrix, random_matrix, T(1), A);
    for (int i = 0; i < n; i++) {
        htool::underlying_type<T> row_sum = 0;
        for (int j = 0; j < n; j++) {
            if (j != i) {
                row_sum += std::abs(A(i, j));
            }
        }
        A(i, i) += row_sum;
    }
    LLtA = A;
    UtUA = A;

    if (side == 'L') {
        add_matrix_matrix_product('N', 'N', T(1) / alpha, A, result, T(0), B);
    } else {
        add_matrix_matrix_product('N', 'N', T(1) / alpha, result, A, T(0), B);
    }

    cholesky_factorization('L', LLtA);
    cholesky_factorization('U', UtUA);

    test_factorization = LLtA;
    test_solve         = B;
    triangular_matrix_matrix_solve(side, 'L', side == 'L' ? 'N' : 'C', 'N', alpha, test_factorization, test_solve);
    triangular_matrix_matrix_solve(side, 'L', side == 'L' ? 'C' : 'N', 'N', T(1.), test_factorization, test_solve);
    error    = normFrob(result - test_solve) / normFrob(result);
    is_error = is_error || !(error < 1e-9);
    cout << "> Errors on lower cholesky matrix matrix solve: " << error << '\n';

    test_factorization = UtUA;
    test_solve         = B;
    triangular_matrix_matrix_solve(side, 'U', side == 'L' ? 'C' : 'N', 'N', alpha, test_factorization, test_solve);
    triangular_matrix_matrix_solve(side, 'U', side == 'L' ? 'N' : 'C', 'N', T(1.), test_factorization, test_solve);
    error    = normFrob(result - test_solve) / normFrob(result);
    is_error = is_error || !(error < 1e-9);
    cout << "> Errors on upper cholesky matrix matrix solve: " << error << '\n';

    return is_error;
}

// well_conditioned: diagonally-dominant A (tight tolerance) vs weakly-boosted (forces 2x2 pivots).
template <typename T>
bool test_symmetric_ldlt_matrix_triangular_solve(int n, int nrhs, char side, bool well_conditioned) {

    bool is_error = false;

    htool::underlying_type<T> error;
    T alpha;
    Matrix<T> result(n, nrhs), B(n, nrhs);
    if (side == 'R') {
        result.resize(nrhs, n);
        B.resize(nrhs, n);
    }
    generate_random_array(result.data(), result.nb_rows() * result.nb_cols());
    generate_random_scalar(alpha);

    Matrix<T> A(n, n), LDLtA, UDUtA, test_factorization, test_solve;
    Matrix<T> random_matrix(n, n);
    generate_random_array(random_matrix.data(), random_matrix.nb_rows() * random_matrix.nb_cols());

    add_matrix_matrix_product('T', 'N', T(1), random_matrix, random_matrix, T(1), A);
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
    LDLtA = A;
    UDUtA = A;

    if (side == 'L') {
        add_matrix_matrix_product('N', 'N', T(1) / alpha, A, result, T(0), B);
    } else {
        add_matrix_matrix_product('N', 'N', T(1) / alpha, result, A, T(0), B);
    }

    ldlt_factorization('S', 'L', LDLtA);
    ldlt_factorization('S', 'U', UDUtA);

    if (!well_conditioned && htool::is_complex<T>()) {
        bool lower_has_2x2 = false, upper_has_2x2 = false;
        for (auto v : LDLtA.get_pivots()) {
            lower_has_2x2 = lower_has_2x2 || v < 0;
        }
        for (auto v : UDUtA.get_pivots()) {
            upper_has_2x2 = upper_has_2x2 || v < 0;
        }
        if (!lower_has_2x2 || !upper_has_2x2) {
            is_error = true;                                                                                                                  // LCOV_EXCL_LINE
            cout << "> WARNING: ill-conditioned symmetric ldlt never triggered a 2x2 Bunch-Kaufman pivot - lost 2x2 branch coverage" << '\n'; // LCOV_EXCL_LINE
        }
    }

    test_factorization = LDLtA;
    test_solve         = B;
    triangular_ldlt_matrix_matrix_solve(side, 'L', side == 'L' ? 'N' : 'T', alpha, test_factorization, test_solve);
    apply_ldlt_diagonal('S', side, 'L', test_factorization, test_solve);
    triangular_ldlt_matrix_matrix_solve(side, 'L', side == 'L' ? 'T' : 'N', T(1.), test_factorization, test_solve);
    error    = normFrob(result - test_solve) / normFrob(result);
    is_error = is_error || !(error < tol);
    cout << "> Errors on lower symmetric ldlt matrix matrix solve: " << error << '\n';

    test_factorization = UDUtA;
    test_solve         = B;
    triangular_ldlt_matrix_matrix_solve(side, 'U', side == 'L' ? 'N' : 'T', alpha, test_factorization, test_solve);
    apply_ldlt_diagonal('S', side, 'U', test_factorization, test_solve);
    triangular_ldlt_matrix_matrix_solve(side, 'U', side == 'L' ? 'T' : 'N', T(1.), test_factorization, test_solve);
    error    = normFrob(result - test_solve) / normFrob(result);
    is_error = is_error || !(error < tol);
    cout << "> Errors on upper symmetric ldlt matrix matrix solve: " << error << '\n';

    return is_error;
}

// well_conditioned: Hermitian PSD A (never needs a 2x2 pivot) vs indefinite M+M^H (forces them).
template <typename T>
bool test_hermitian_ldlt_matrix_triangular_solve(int n, int nrhs, char side, bool well_conditioned) {

    bool is_error = false;

    htool::underlying_type<T> error;
    T alpha;
    Matrix<T> result(n, nrhs), B(n, nrhs);
    if (side == 'R') {
        result.resize(nrhs, n);
        B.resize(nrhs, n);
    }
    generate_random_array(result.data(), result.nb_rows() * result.nb_cols());
    generate_random_scalar(alpha);

    Matrix<T> A(n, n), LDLtA, UDUtA, test_factorization, test_solve;
    htool::underlying_type<T> tol;
    if (well_conditioned) {
        Matrix<T> random_matrix(n, n);
        generate_random_array(random_matrix.data(), random_matrix.nb_rows() * random_matrix.nb_cols());
        add_matrix_matrix_product('C', 'N', T(1), random_matrix, random_matrix, T(1), A);
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
        Matrix<T> M(n, n);
        generate_random_array(M.data(), M.nb_rows() * M.nb_cols());
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                A(i, j) = M(i, j) + conj_if_complex(M(j, i));
            }
        }
        tol = 1e-6;
    }
    LDLtA = A;
    UDUtA = A;

    if (side == 'L') {
        add_matrix_matrix_product('N', 'N', T(1) / alpha, A, result, T(0), B);
    } else {
        add_matrix_matrix_product('N', 'N', T(1) / alpha, result, A, T(0), B);
    }

    ldlt_factorization('H', 'L', LDLtA);
    ldlt_factorization('H', 'U', UDUtA);

    if (!well_conditioned && htool::is_complex<T>()) {
        bool lower_has_2x2 = false, upper_has_2x2 = false;
        for (auto v : LDLtA.get_pivots()) {
            lower_has_2x2 = lower_has_2x2 || v < 0;
        }
        for (auto v : UDUtA.get_pivots()) {
            upper_has_2x2 = upper_has_2x2 || v < 0;
        }
        if (!lower_has_2x2 || !upper_has_2x2) {
            is_error = true;                                                                                                             // LCOV_EXCL_LINE
            cout << "> WARNING: indefinite hermitian ldlt never triggered a 2x2 Bunch-Kaufman pivot - lost 2x2 branch coverage" << '\n'; // LCOV_EXCL_LINE
        }
    }

    test_factorization = LDLtA;
    test_solve         = B;
    triangular_ldlt_matrix_matrix_solve(side, 'L', side == 'L' ? 'N' : 'C', alpha, test_factorization, test_solve);
    apply_ldlt_diagonal('H', side, 'L', test_factorization, test_solve);
    triangular_ldlt_matrix_matrix_solve(side, 'L', side == 'L' ? 'C' : 'N', T(1.), test_factorization, test_solve);
    error    = normFrob(result - test_solve) / normFrob(result);
    is_error = is_error || !(error < tol);
    cout << "> Errors on lower hermitian ldlt matrix matrix solve: " << error << '\n';

    test_factorization = UDUtA;
    test_solve         = B;
    triangular_ldlt_matrix_matrix_solve(side, 'U', side == 'L' ? 'N' : 'C', alpha, test_factorization, test_solve);
    apply_ldlt_diagonal('H', side, 'U', test_factorization, test_solve);
    triangular_ldlt_matrix_matrix_solve(side, 'U', side == 'L' ? 'C' : 'N', T(1.), test_factorization, test_solve);
    error    = normFrob(result - test_solve) / normFrob(result);
    is_error = is_error || !(error < tol);
    cout << "> Errors on upper hermitian ldlt matrix matrix solve: " << error << '\n';

    return is_error;
}
