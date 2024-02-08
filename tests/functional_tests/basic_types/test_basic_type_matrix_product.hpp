#include <htool/basic_types/matrix.hpp>
#include <htool/testing/generator_input.hpp>

using namespace std;
using namespace htool;

template <typename T>
bool test_matrix_product(int nr, int nc, int mu, char op, char symmetry, char UPLO) {

    bool is_error = false;

    // Generate random matrix
    vector<T> data(nr * nc);
    generate_random_vector(data);
    Matrix<T> matrix;
    matrix.assign(nr, nc, data.data(), false);

    if (symmetry == 'S') {
        for (int i = 0; i < nr; i++) {
            for (int j = 0; j < i; j++) {
                matrix(i, j) = matrix(j, i);
            }
        }
    } else if (symmetry == 'H') {
        for (int i = 0; i < nr; i++) {
            for (int j = 0; j < i; j++) {
                matrix(i, j) = conj_if_complex(matrix(j, i));
            }
            matrix(i, i) = std::real(matrix(i, i));
        }
    }

    // Input
    int ni = (op == 'T' || op == 'C') ? nr : nc;
    int no = (op == 'T' || op == 'C') ? nc : nr;
    vector<T> x(ni * mu, 1), y(no * mu, 1), ref(no * mu, 0), out(ref);
    T alpha, beta;
    htool::underlying_type<T> error;
    generate_random_vector(x);
    generate_random_vector(y);
    generate_random_scalar(alpha);
    generate_random_scalar(beta);

    // reference
    if (op == 'N') {
        for (int p = 0; p < mu; p++) {
            for (int i = 0; i < no; i++) {
                for (int j = 0; j < ni; j++) {
                    ref.at(i + p * no) += alpha * matrix(i, j) * x.at(j + p * ni);
                }
                ref.at(i + p * no) += beta * y.at(i + p * no);
            }
        }
    } else if (op == 'T') {
        for (int p = 0; p < mu; p++) {
            for (int i = 0; i < no; i++) {
                for (int j = 0; j < ni; j++) {
                    ref.at(i + p * no) += alpha * matrix(j, i) * x.at(j + p * ni);
                }
                ref.at(i + p * no) += beta * y.at(i + p * no);
            }
        }
    } else if (op == 'C') {
        for (int p = 0; p < mu; p++) {
            for (int i = 0; i < no; i++) {
                for (int j = 0; j < ni; j++) {
                    ref.at(i + p * no) += alpha * conj_if_complex(matrix(j, i)) * x.at(j + p * ni);
                }
                ref.at(i + p * no) += beta * y.at(i + p * no);
            }
        }
    }

    // Row major inputs
    vector<T> x_row_major(ni * mu, 1), y_row_major(no * mu, 1), ref_row_major(no * mu, 0);
    for (int i = 0; i < ni; i++) {
        for (int j = 0; j < mu; j++) {
            x_row_major[i * mu + j] = x[i + j * ni];
        }
    }

    for (int i = 0; i < no; i++) {
        for (int j = 0; j < mu; j++) {
            y_row_major[i * mu + j]   = y[i + j * no];
            ref_row_major[i * mu + j] = ref[i + j * no];
        }
    }

    // Product
    if (mu == 1) {
        out = y;
        matrix.add_vector_product(op, alpha, x.data(), beta, out.data());
        error    = norm2(ref - out) / norm2(ref);
        is_error = is_error || !(error < 1e-14);
        cout << "> Errors on a matrix vector product: " << error << endl;
    }

    if (mu == 1 && symmetry != 'N') {
        out = y;
        matrix.add_vector_product_symmetric(op, alpha, x.data(), beta, out.data(), UPLO, symmetry);
        error    = norm2(ref - out) / norm2(ref);
        is_error = is_error || !(error < 1e-14);
        cout << "> Errors on a symmetric matrix vector product: " << error << endl;
    }

    out = y;
    matrix.add_matrix_product(op, alpha, x.data(), beta, out.data(), mu);
    error    = norm2(ref - out) / norm2(ref);
    is_error = is_error || !(error < 1e-14);
    cout << "> Errors on a matrix matrix product: " << error << endl;

    out = y_row_major;
    matrix.add_matrix_product_row_major(op, alpha, x_row_major.data(), beta, out.data(), mu);
    error    = norm2(ref_row_major - out) / norm2(ref_row_major);
    is_error = is_error || !(error < 1e-14);
    cout << "> Errors on a matrix matrix product with row major input: " << error << endl;

    if (symmetry != 'N') {
        out = y;
        matrix.add_matrix_product_symmetric(op, alpha, x.data(), beta, out.data(), mu, UPLO, symmetry);
        error    = norm2(ref - out) / norm2(ref);
        is_error = is_error || !(error < 1e-14);
        cout << "> Errors on a symmetric matrix matrix product: " << error << endl;

        out = y_row_major;
        matrix.add_matrix_product_symmetric_row_major(op, alpha, x_row_major.data(), beta, out.data(), mu, UPLO, symmetry);
        error    = norm2(ref_row_major - out) / norm2(ref_row_major);
        is_error = is_error || !(error < 1e-14);
        cout << "> Errors on a symmetric matrix matrix product with row major input: " << error << endl;
    }
    return is_error;
}
