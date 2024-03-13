#include <htool/matrix/linalg/interface.hpp>
#include <htool/matrix/matrix.hpp>
#include <htool/testing/generator_input.hpp>

using namespace std;
using namespace htool;

template <typename T>
bool test_matrix_product(int nr, int nc, int mu, char transa, char symmetry, char UPLO) {

    bool is_error = false;

    // Generate random matrix
    vector<T> data(nr * nc);
    generate_random_vector(data);
    Matrix<T> A;
    A.assign(nr, nc, data.data(), false);

    if (symmetry == 'S') {
        for (int i = 0; i < nr; i++) {
            for (int j = 0; j < i; j++) {
                A(i, j) = A(j, i);
            }
        }
    } else if (symmetry == 'H') {
        for (int i = 0; i < nr; i++) {
            for (int j = 0; j < i; j++) {
                A(i, j) = conj_if_complex(A(j, i));
            }
            A(i, i) = std::real(A(i, i));
        }
    }

    // Input sizes
    int ni = (transa == 'T' || transa == 'C') ? nr : nc;
    int no = (transa == 'T' || transa == 'C') ? nc : nr;

    // Random input matrices
    Matrix<T> X(ni, mu), Y(no, mu), Xt(mu, ni), Yt(mu, no), C;
    T alpha, beta, scaling_coefficient;
    htool::underlying_type<T> error;
    generate_random_array(X.data(), X.nb_cols() * X.nb_rows());
    generate_random_array(Y.data(), Y.nb_cols() * Y.nb_rows());
    generate_random_scalar(alpha);
    generate_random_scalar(beta);
    generate_random_scalar(scaling_coefficient);
    transpose(X, Xt);
    transpose(Y, Yt);

    // reference
    Matrix<T> matrix_result_w_matrix_sum(no, mu), transposed_matrix_result_w_matrix_sum;
    if (transa == 'N') {
        for (int p = 0; p < mu; p++) {
            for (int i = 0; i < no; i++) {
                for (int j = 0; j < ni; j++) {
                    matrix_result_w_matrix_sum(i, p) += alpha * A(i, j) * X(j, p);
                }
                matrix_result_w_matrix_sum(i, p) += beta * Y(i, p);
            }
        }
    } else if (transa == 'T') {
        for (int p = 0; p < mu; p++) {
            for (int i = 0; i < no; i++) {
                for (int j = 0; j < ni; j++) {
                    matrix_result_w_matrix_sum(i, p) += alpha * A(j, i) * X(j, p);
                }
                matrix_result_w_matrix_sum(i, p) += beta * Y(i, p);
            }
        }
    } else if (transa == 'C') {
        for (int p = 0; p < mu; p++) {
            for (int i = 0; i < no; i++) {
                for (int j = 0; j < ni; j++) {
                    matrix_result_w_matrix_sum(i, p) += alpha * conj_if_complex(A(j, i)) * X(j, p);
                }
                matrix_result_w_matrix_sum(i, p) += beta * Y(i, p);
            }
        }
    }
    transpose(matrix_result_w_matrix_sum, transposed_matrix_result_w_matrix_sum);
    Matrix<T> scaled_matrix_result_w_matrix_sum(matrix_result_w_matrix_sum);
    scale(scaling_coefficient, scaled_matrix_result_w_matrix_sum);

    // Product
    if (mu == 1) {
        C = Y;
        add_matrix_vector_product(transa, alpha, A, X.data(), beta, C.data());
        error    = normFrob(matrix_result_w_matrix_sum - C) / normFrob(matrix_result_w_matrix_sum);
        is_error = is_error || !(error < 1e-14);
        cout << "> Errors on a matrix vector product: " << error << endl;
    }

    if (mu == 1 && symmetry != 'N') {
        C = Y;
        add_matrix_vector_product_symmetric(transa, alpha, A, X.data(), beta, C.data(), UPLO, symmetry);
        error    = normFrob(matrix_result_w_matrix_sum - C) / normFrob(matrix_result_w_matrix_sum);
        is_error = is_error || !(error < 1e-14);
        cout << "> Errors on a symmetric matrix vector product: " << error << endl;
    }

    C = Y;
    add_matrix_matrix_product(transa, 'N', alpha, A, X, beta, C);
    error    = normFrob(matrix_result_w_matrix_sum - C) / normFrob(matrix_result_w_matrix_sum);
    is_error = is_error || !(error < 1e-14);
    cout << "> Errors on a matrix matrix product: " << error << endl;

    C = Yt;
    add_matrix_matrix_product_row_major(transa, 'N', alpha, A, Xt.data(), beta, C.data(), mu);
    error    = normFrob(transposed_matrix_result_w_matrix_sum - C) / normFrob(transposed_matrix_result_w_matrix_sum);
    is_error = is_error || !(error < 1e-14);
    cout << "> Errors on a matrix matrix product with row major input: " << error << endl;

    if (symmetry != 'N') {
        C = Y;
        add_matrix_matrix_product_symmetric('L', transa, 'N', alpha, A, X, beta, C, UPLO, symmetry);
        error    = normFrob(matrix_result_w_matrix_sum - C) / normFrob(matrix_result_w_matrix_sum);
        is_error = is_error || !(error < 1e-14);
        cout << "> Errors on a symmetric matrix matrix product: " << error << endl;

        C = Yt;
        add_matrix_matrix_product_symmetric_row_major('L', transa, 'N', alpha, A, Xt.data(), beta, C.data(), mu, UPLO, symmetry);
        error    = normFrob(transposed_matrix_result_w_matrix_sum - C) / normFrob(transposed_matrix_result_w_matrix_sum);
        is_error = is_error || !(error < 1e-14);
        cout << "> Errors on a symmetric matrix matrix product with row major input: " << error << endl;
    }

    C                           = Y;
    T local_scaling_coefficient = scaling_coefficient;
    if (transa == 'C') {
        local_scaling_coefficient = conj_if_complex(scaling_coefficient);
    }
    scale(local_scaling_coefficient, A);
    add_matrix_matrix_product(transa, 'N', alpha, A, X, scaling_coefficient * beta, C);
    error    = normFrob(scaled_matrix_result_w_matrix_sum - C) / normFrob(scaled_matrix_result_w_matrix_sum);
    is_error = is_error || !(error < 1e-14);
    cout << "> Errors on a scaled matrix matrix product: " << error << endl;

    return is_error;
}
