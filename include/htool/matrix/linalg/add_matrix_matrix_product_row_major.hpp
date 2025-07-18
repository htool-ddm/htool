#ifndef HTOOL_MATRIX_LINALG_ADD_MATRIX_MATRIX_PRODUCT_ROW_MAJOR_HPP
#define HTOOL_MATRIX_LINALG_ADD_MATRIX_MATRIX_PRODUCT_ROW_MAJOR_HPP

#include "../../matrix/matrix.hpp"         // for Matrix
#include "../../misc/misc.hpp"             // for conj_if_complex
#include "../../wrappers/wrapper_blas.hpp" // for Blas
#include <algorithm>                       // for transform
#include <complex>                         // for complex, conj
#include <vector>                          // for vector
namespace htool {

/// @brief It computes C := alpha*op( A )*op( B ) + beta*C from B**T and C**T using C**T := alpha*op( B**T )*op( A**T ) + beta*C**T
/// @tparam T
/// @param transa
/// @param transb
/// @param alpha
/// @param A
/// @param in
/// @param beta
/// @param out
/// @param mu
template <typename T>
void add_matrix_matrix_product_row_major(char transa, char transb, T alpha, const Matrix<T> &A, const T *in, T beta, T *out, int mu) {

    int nr               = A.nb_rows();
    int nc               = A.nb_cols();
    char inverted_transa = 'N';
    char inverted_transb = 'T';
    int M                = mu;
    int N                = nr;
    int K                = nc;
    if (transb != 'N') {
        inverted_transa = 'T';
    }
    if (transa != 'N') {
        inverted_transb = 'N';
        N               = nc;
        K               = nr;
    }

    int lda = inverted_transa == 'N' ? M : K;
    int ldb = inverted_transb == 'N' ? K : N;
    int ldc = M;

    Blas<T>::gemm(&inverted_transa, &inverted_transb, &M, &N, &K, &alpha, in, &lda, A.data(), &ldb, &beta, out, &ldc);
}

template <typename T>
void add_matrix_matrix_product_row_major(char transa, char transb, std::complex<T> alpha, const Matrix<std::complex<T>> &A, const std::complex<T> *in, std::complex<T> beta, std::complex<T> *out, int mu) {

    int nr               = A.nb_rows();
    int nc               = A.nb_cols();
    char inverted_transa = 'N';
    char inverted_transb = 'T';
    int M                = mu;
    int N                = nr;
    int K                = nc;
    if (transb != 'N') {
        inverted_transa = 'T';
    }
    if (transa != 'N') {
        inverted_transb = 'N';
        N               = nc;
        K               = nr;
    }

    int lda = inverted_transa == 'N' ? M : K;
    int ldb = inverted_transb == 'N' ? K : N;
    int ldc = M;

    std::vector<std::complex<T>> buffer_A(transa == 'C' ? nr * nc : 0);
    std::vector<std::complex<T>> buffer_B(transb == 'C' ? M * lda : 0);

    if (transa == 'C') {
        std::transform(A.data(), A.data() + nr * nc, buffer_A.data(), [](const std::complex<T> &c) { return std::conj(c); });
    }
    if (transb == 'C') {
        std::transform(in, in + M * lda, buffer_B.data(), [](const std::complex<T> &c) { return std::conj(c); });
    }

    const std::complex<T> *B_final = transb != 'C' ? in : buffer_B.data();
    const std::complex<T> *A_final = transa != 'C' ? A.data() : buffer_A.data();
    Blas<std::complex<T>>::gemm(&inverted_transa, &inverted_transb, &M, &N, &K, &alpha, B_final, &lda, A_final, &ldb, &beta, out, &ldc);
}

template <typename T>
void add_symmetric_matrix_matrix_product_row_major(char side, char UPLO, T alpha, const Matrix<T> &A, const T *in, T beta, T *out, const int &mu) {

    int nr = A.nb_rows();
    if (nr) {
        char inverted_side = 'R';
        int M              = mu;
        int N              = nr;
        int lda            = N;
        if (side == 'R') {
            inverted_side = 'L';
            M             = nr;
            N             = mu;
            lda           = M;
        }
        int ldb = M;
        int ldc = M;

        Blas<T>::symm(&inverted_side, &UPLO, &M, &N, &alpha, A.data(), &lda, in, &ldb, &beta, out, &ldc);
    }
}

template <typename T>
void add_hermitian_matrix_matrix_product_row_major(char side, char UPLO, T alpha, const Matrix<T> &A, const T *in, T beta, T *out, const int &mu) {
    add_symmetric_matrix_matrix_product_row_major(side, UPLO, alpha, A, in, beta, out, mu);
}

template <typename T>
void add_hermitian_matrix_matrix_product_row_major(char side, char UPLO, std::complex<T> alpha, const Matrix<std::complex<T>> &A, const std::complex<T> *in, std::complex<T> beta, std::complex<T> *out, const int &mu) {

    int nr = A.nb_rows();
    if (nr) {
        char inverted_side = 'R';
        int M              = mu;
        int N              = nr;
        int lda            = N;
        if (side == 'R') {
            inverted_side = 'L';
            M             = nr;
            N             = mu;
            lda           = M;
        }
        int ldb = M;
        int ldc = M;

        std::vector<std::complex<T>> conjugate_in(nr * mu);
        std::complex<T> conjugate_alpha = std::conj(alpha);
        std::complex<T> conjugate_beta  = std::conj(beta);
        std::transform(in, in + nr * mu, conjugate_in.data(), [](const std::complex<T> &c) { return std::conj(c); });
        conj_if_complex<std::complex<T>>(out, A.nb_cols() * mu);
        Blas<std::complex<T>>::hemm(&inverted_side, &UPLO, &M, &N, &conjugate_alpha, A.data(), &lda, conjugate_in.data(), &ldb, &conjugate_beta, out, &ldc);
        conj_if_complex<std::complex<T>>(out, A.nb_cols() * mu);
    }
}

} // namespace htool

#endif
