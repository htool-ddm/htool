#ifndef HTOOL_MATRIX_LINALG_FACTORIZATION_HPP
#define HTOOL_MATRIX_LINALG_FACTORIZATION_HPP

#include "../../matrix/matrix.hpp"           // for Matrix
#include "../../wrappers/wrapper_blas.hpp"   // for Blas
#include "../../wrappers/wrapper_lapack.hpp" // for Lapack
#include <algorithm>                         // for min

namespace htool {

template <typename T>
void lu_factorization(Matrix<T> &A) {
    int M      = A.nb_rows();
    int N      = A.nb_cols();
    int lda    = M;
    auto &ipiv = A.get_pivots();
    ipiv.resize(std::min(M, N));
    int info;

    Lapack<T>::getrf(&M, &N, A.data(), &lda, ipiv.data(), &info);
}

template <typename T>
void triangular_matrix_matrix_solve(char side, char UPLO, char transa, char diag, T alpha, const Matrix<T> &A, Matrix<T> &B) {
    int m           = B.nb_rows();
    int n           = B.nb_cols();
    int lda         = side == 'L' ? m : n;
    int ldb         = m;
    auto &ipiv      = A.get_pivots();
    bool is_pivoted = false;

    if (ipiv.size() > 0) {
        int index = 0;
        while (index < ipiv.size() and not is_pivoted) {
            is_pivoted = !(ipiv[index] == index + 1);
            index += 1;
        }
    }

    if (is_pivoted and UPLO == 'L') {
        if (side == 'L' and transa == 'N') {
            int K1   = 1;
            int K2   = m;
            int incx = 1;
            Lapack<T>::laswp(&n, B.data(), &ldb, &K1, &K2, ipiv.data(), &incx);
        } else if (side == 'R' and transa != 'N') {
            int incx = 1;
            int incy = 1;
            for (int j = 0; j < B.nb_cols(); j++) {
                Blas<T>::swap(&m, &B(0, ipiv[j] - 1), &incx, &B(0, j), &incy);
            }
        }
    }

    Blas<T>::trsm(&side, &UPLO, &transa, &diag, &m, &n, &alpha, A.data(), &lda, B.data(), &ldb);

    if (is_pivoted and UPLO == 'L') {
        if (side == 'L' and transa != 'N') {
            int K1   = 1;
            int K2   = m;
            int incx = -1;
            Lapack<T>::laswp(&n, B.data(), &ldb, &K1, &K2, ipiv.data(), &incx);
        } else if (side == 'R' and transa == 'N') {
            int incx = 1;
            int incy = 1;
            for (int j = B.nb_cols() - 1; j >= 0; j--) {
                Blas<T>::swap(&m, &B(0, ipiv[j] - 1), &incx, &B(0, j), &incy);
            }
        }
    }
}

template <typename T>
void lu_solve(char trans, const Matrix<T> &A, Matrix<T> &B) {
    int M      = A.nb_rows();
    int NRHS   = B.nb_cols();
    int lda    = M;
    int ldb    = M;
    auto &ipiv = A.get_pivots();
    int info;

    Lapack<T>::getrs(&trans, &M, &NRHS, A.data(), &lda, ipiv.data(), B.data(), &ldb, &info);
}

template <typename T>
void cholesky_factorization(char UPLO, Matrix<T> &A) {
    int M   = A.nb_rows();
    int lda = M;
    int info;

    Lapack<T>::potrf(&UPLO, &M, A.data(), &lda, &info);
}

template <typename T>
void cholesky_solve(char UPLO, const Matrix<T> &A, Matrix<T> &B) {
    int M    = A.nb_rows();
    int NRHS = B.nb_cols();
    int lda  = M;
    int ldb  = M;
    int info;

    Lapack<T>::potrs(&UPLO, &M, &NRHS, A.data(), &lda, B.data(), &ldb, &info);
}

} // namespace htool
#endif
