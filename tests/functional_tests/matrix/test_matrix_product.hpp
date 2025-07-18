#include <complex>
#include <htool/matrix/linalg/add_matrix_matrix_product.hpp>
#include <htool/matrix/linalg/add_matrix_matrix_product_row_major.hpp>
#include <htool/matrix/linalg/add_matrix_vector_product.hpp>
#include <htool/matrix/linalg/scale.hpp>
#include <htool/matrix/linalg/transpose.hpp>
#include <htool/matrix/matrix.hpp>
#include <htool/misc/misc.hpp>
#include <htool/testing/generator_input.hpp>
#include <iostream>

using namespace std;
using namespace htool;

template <typename T>
bool test_matrix_product(int n1, int n2, int n3, char transa, char transb) {

    bool is_error = false;

    // Generate random matrices
    Matrix<T> A, B, C(n1, n3), Y(C), Yt(n3, n1);
    if (transa == 'N') {
        A.resize(n1, n2);
    } else {
        A.resize(n2, n1);
    }
    if (transb == 'N') {
        B.resize(n2, n3);
    } else {
        B.resize(n3, n2);
    }
    Matrix<T> Bt(B.nb_cols(), B.nb_rows());
    generate_random_array(A.data(), A.nb_cols() * A.nb_rows());
    generate_random_array(B.data(), B.nb_cols() * B.nb_rows());
    generate_random_array(Y.data(), Y.nb_cols() * Y.nb_rows());
    transpose(B, Bt);
    transpose(Y, Yt);

    // Random input
    T alpha, beta, scaling_coefficient;
    htool::underlying_type<T> error;
    generate_random_scalar(alpha);
    generate_random_scalar(beta);
    generate_random_scalar(scaling_coefficient);

    // reference
    Matrix<T> matrix_result_w_matrix_sum(n1, n3), transposed_matrix_result_w_matrix_sum(n3, n1);
    for (int p = 0; p < n3; p++) {
        for (int i = 0; i < n1; i++) {
            for (int j = 0; j < n2; j++) {
                int A_left_index  = transa == 'N' ? i : j;
                int A_right_index = transa == 'N' ? j : i;

                int B_left_index  = transb == 'N' ? j : p;
                int B_right_index = transb == 'N' ? p : j;

                auto A_coef = A(A_left_index, A_right_index);
                auto B_coef = B(B_left_index, B_right_index);
                if (transa == 'C') {
                    A_coef = conj_if_complex(A_coef);
                }
                if (transb == 'C') {
                    B_coef = conj_if_complex(B_coef);
                }
                matrix_result_w_matrix_sum(i, p) += alpha * A_coef * B_coef;
            }
            matrix_result_w_matrix_sum(i, p) += beta * Y(i, p);
        }
    }

    transpose(matrix_result_w_matrix_sum, transposed_matrix_result_w_matrix_sum);
    Matrix<T> scaled_matrix_result_w_matrix_sum(matrix_result_w_matrix_sum);
    scale(scaling_coefficient, scaled_matrix_result_w_matrix_sum);

    // Product
    if (n3 == 1 && transb != 'C') {
        C = Y;
        add_matrix_vector_product(transa, alpha, A, B.data(), beta, C.data());
        error    = normFrob(matrix_result_w_matrix_sum - C) / normFrob(matrix_result_w_matrix_sum);
        is_error = is_error || !(error < 1e-14);
        cout << "> Errors on a matrix vector product: " << error << endl;
    }

    C = Y;
    add_matrix_matrix_product(transa, transb, alpha, A, B, beta, C);
    error    = normFrob(matrix_result_w_matrix_sum - C) / normFrob(matrix_result_w_matrix_sum);
    is_error = is_error || !(error < 1e-14);
    cout << "> Errors on a matrix matrix product: " << error << endl;

    C = Yt;
    add_matrix_matrix_product_row_major(transa, transb, alpha, A, Bt.data(), beta, C.data(), n3);
    error    = normFrob(transposed_matrix_result_w_matrix_sum - C) / normFrob(transposed_matrix_result_w_matrix_sum);
    is_error = is_error || !(error < 1e-14);
    cout << "> Errors on a matrix matrix product with row major input: " << error << endl;

    C                           = Y;
    T local_scaling_coefficient = scaling_coefficient;
    if (transa == 'C') {
        local_scaling_coefficient = conj_if_complex(scaling_coefficient);
    }
    scale(local_scaling_coefficient, A);
    add_matrix_matrix_product(transa, transb, alpha, A, B, scaling_coefficient * beta, C);
    error    = normFrob(scaled_matrix_result_w_matrix_sum - C) / normFrob(scaled_matrix_result_w_matrix_sum);
    is_error = is_error || !(error < 1e-14);
    cout << "> Errors on a scaled matrix matrix product: " << error << endl;

    return is_error;
}

template <typename T>
bool test_matrix_symmetric_product(int n1, int n2, char side, char UPLO) {

    bool is_error = false;

    // Generate random matrices
    Matrix<T> A(n1, n1), B(n1, n2), C(n1, n2), Y(C), Yt(n2, n1);
    if (side != 'L') {
        B.resize(n2, n1);
        C.resize(n2, n1);
        Y = C;
        Yt.resize(n1, n2);
    }
    Matrix<T> Bt(B.nb_cols(), B.nb_rows());
    generate_random_array(A.data(), A.nb_cols() * A.nb_rows());
    generate_random_array(B.data(), B.nb_cols() * B.nb_rows());
    generate_random_array(Y.data(), Y.nb_cols() * Y.nb_rows());
    transpose(B, Bt);
    transpose(Y, Yt);

    for (int i = 0; i < n1; i++) {
        for (int j = 0; j < i; j++) {
            A(i, j) = A(j, i);
        }
    }

    // Random input matrices
    T alpha(1), beta(0), scaling_coefficient;
    htool::underlying_type<T> error;
    generate_random_scalar(alpha);
    generate_random_scalar(beta);
    generate_random_scalar(scaling_coefficient);

    // reference
    Matrix<T> matrix_result_w_matrix_sum(Y);
    Matrix<T> transposed_matrix_result_w_matrix_sum(matrix_result_w_matrix_sum.nb_cols(), matrix_result_w_matrix_sum.nb_rows());

    if (side == 'L') {
        add_matrix_matrix_product('N', 'N', alpha, A, B, beta, matrix_result_w_matrix_sum);
    } else {
        add_matrix_matrix_product('N', 'N', alpha, B, A, beta, matrix_result_w_matrix_sum);
    }
    transpose(matrix_result_w_matrix_sum, transposed_matrix_result_w_matrix_sum);
    Matrix<T> scaled_matrix_result_w_matrix_sum(matrix_result_w_matrix_sum);
    scale(scaling_coefficient, scaled_matrix_result_w_matrix_sum);

    if (n2 == 1) {
        C = Y;
        add_symmetric_matrix_vector_product(UPLO, alpha, A, B.data(), beta, C.data());
        error    = normFrob(matrix_result_w_matrix_sum - C) / normFrob(matrix_result_w_matrix_sum);
        is_error = is_error || !(error < 1e-14);
        cout << "> Errors on a symmetry matrix vector product: " << error << endl;

        if constexpr (!is_complex<T>()) {
            C = Y;
            add_hermitian_matrix_vector_product(UPLO, alpha, A, B.data(), beta, C.data());
            error    = normFrob(matrix_result_w_matrix_sum - C) / normFrob(matrix_result_w_matrix_sum);
            is_error = is_error || !(error < 1e-14);
            cout << "> Errors on a hermitian matrix vector product: " << error << endl;
        }
    }
    C = Y;
    add_symmetric_matrix_matrix_product(side, UPLO, alpha, A, B, beta, C);
    error    = normFrob(matrix_result_w_matrix_sum - C) / normFrob(matrix_result_w_matrix_sum);
    is_error = is_error || !(error < 1e-14);
    cout << "> Errors on a symmetric matrix matrix product: " << error << endl;

    C = Yt;
    add_symmetric_matrix_matrix_product_row_major(side, UPLO, alpha, A, Bt.data(), beta, C.data(), n2);
    error    = normFrob(transposed_matrix_result_w_matrix_sum - C) / normFrob(transposed_matrix_result_w_matrix_sum);
    is_error = is_error || !(error < 1e-14);
    cout << "> Errors on a symmetric matrix matrix product with row major input: " << error << endl;

    if constexpr (!is_complex<T>()) {
        C = Y;
        add_hermitian_matrix_matrix_product(side, UPLO, alpha, A, B, beta, C);
        error    = normFrob(matrix_result_w_matrix_sum - C) / normFrob(matrix_result_w_matrix_sum);
        is_error = is_error || !(error < 1e-14);
        cout << "> Errors on a hermitian matrix matrix product: " << error << endl;

        C = Yt;
        add_hermitian_matrix_matrix_product_row_major(side, UPLO, alpha, A, Bt.data(), beta, C.data(), n2);
        error    = normFrob(transposed_matrix_result_w_matrix_sum - C) / normFrob(transposed_matrix_result_w_matrix_sum);
        is_error = is_error || !(error < 1e-14);
        cout << "> Errors on a hermitian matrix matrix product with row major input: " << error << endl;
    }

    return is_error;
}

template <typename T>
bool test_matrix_hermitian_product(int n1, int n2, char side, char UPLO) {

    bool is_error = false;

    // Generate random matrices
    Matrix<std::complex<T>> A(n1, n1), B(n1, n2), C(n1, n2), Y(C), Yt(n2, n1);
    if (side != 'L') {
        B.resize(n2, n1);
        C.resize(n2, n1);
        Y = C;
        Yt.resize(n1, n2);
    }
    Matrix<std::complex<T>> Bt(B.nb_cols(), B.nb_rows());
    generate_random_array(A.data(), A.nb_cols() * A.nb_rows());
    generate_random_array(B.data(), B.nb_cols() * B.nb_rows());
    generate_random_array(Y.data(), Y.nb_cols() * Y.nb_rows());
    transpose(B, Bt);
    transpose(Y, Yt);

    for (int i = 0; i < n1; i++) {
        for (int j = 0; j < i; j++) {
            A(i, j) = conj_if_complex(A(j, i));
        }
        A(i, i) = std::real(A(i, i));
    }

    // Random input matrices
    std::complex<T> alpha(1), beta(0), scaling_coefficient;
    T error;
    generate_random_scalar(alpha);
    generate_random_scalar(beta);
    generate_random_scalar(scaling_coefficient);

    // reference
    Matrix<std::complex<T>> matrix_result_w_matrix_sum(Y);
    Matrix<std::complex<T>> transposed_matrix_result_w_matrix_sum(matrix_result_w_matrix_sum.nb_cols(), matrix_result_w_matrix_sum.nb_rows());

    if (side == 'L') {
        add_matrix_matrix_product('N', 'N', alpha, A, B, beta, matrix_result_w_matrix_sum);
    } else {
        add_matrix_matrix_product('N', 'N', alpha, B, A, beta, matrix_result_w_matrix_sum);
    }
    transpose(matrix_result_w_matrix_sum, transposed_matrix_result_w_matrix_sum);
    Matrix<std::complex<T>> scaled_matrix_result_w_matrix_sum(matrix_result_w_matrix_sum);
    scale(scaling_coefficient, scaled_matrix_result_w_matrix_sum);

    if (n2 == 1 && side == 'L') {
        C = Y;
        add_hermitian_matrix_vector_product(UPLO, alpha, A, B.data(), beta, C.data());
        error    = normFrob(matrix_result_w_matrix_sum - C) / normFrob(matrix_result_w_matrix_sum);
        is_error = is_error || !(error < 1e-14);
        cout << "> Errors on a hermitian matrix vector product: " << error << endl;
    }

    C = Y;
    add_hermitian_matrix_matrix_product(side, UPLO, alpha, A, B, beta, C);
    error    = normFrob(matrix_result_w_matrix_sum - C) / normFrob(matrix_result_w_matrix_sum);
    is_error = is_error || !(error < 1e-14);
    cout << "> Errors on a hermitian matrix matrix product: " << error << endl;

    C = Yt;
    add_hermitian_matrix_matrix_product_row_major(side, UPLO, alpha, A, Bt.data(), beta, C.data(), n2);
    error    = normFrob(transposed_matrix_result_w_matrix_sum - C) / normFrob(transposed_matrix_result_w_matrix_sum);
    is_error = is_error || !(error < 1e-14);
    cout << "> Errors on a hermitian matrix matrix product with row major input: " << error << endl;

    return is_error;
}
