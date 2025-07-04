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
            is_pivoted = (ipiv[index] == index + 1);
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
    // std::cout <<"TEST "<<ipiv<<" "<<is_pivoted<<"\n";
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

template <typename T>
void symmetric_ldlt_factorization(char UPLO, Matrix<T> &A) {
    int N      = A.nb_rows();
    int lda    = N;
    auto &ipiv = A.get_pivots();
    ipiv.resize(N);
    std::vector<T> work(1);
    int lwork = -1;
    int info;

    Lapack<T>::sytrf(&UPLO, &N, A.data(), &lda, ipiv.data(), work.data(), &lwork, &info);
    lwork = static_cast<int>(std::real(work[0]));
    work.resize(lwork);
    Lapack<T>::sytrf(&UPLO, &N, A.data(), &lda, ipiv.data(), work.data(), &lwork, &info);
}

template <typename T>
void triangular_ldlt_matrix_matrix_solve(char side, char UPLO, char transa, char diag, T alpha, const Matrix<T> &A, Matrix<T> &B) {
    int M      = A.nb_rows();
    int N      = side == 'L' ? B.nb_cols() : B.nb_rows();
    auto &ipiv = A.get_pivots();
    int lda    = M;
    int ldb    = M;
    int local_size;
    int kp;
    int ione    = 1;
    T one       = 1;
    T minus_one = -1;
    T factor;
    T akm1k, akm1, ak, denom, bkm1, bk;

    if (UPLO == 'L' && transa == 'N' && side == 'L') {
        int k = 0;
        while (k < M) {
            // 1x1 diagonal block
            if (ipiv[k] > 0) {
                // Permute
                kp = ipiv[k] - 1;
                if (kp != k) {
                    Blas<T>::swap(&N, &B(k, 0), &ldb, &B(kp, 0), &ldb);
                }

                // Apply inv(L(k))
                local_size = M - k - 1;
                if (local_size >= 0) {
                    Blas<T>::ger(&local_size, &N, &minus_one, &A(k + 1, k), &ione, &B(k, 0), &ldb, &B(k + 1, 0), &ldb);
                }

                // Inverse of D(k)
                factor = 1. / A(k, k);
                Blas<T>::scal(&N, &factor, &B(k, 0), &ldb);

                // Increment
                k += 1;
            }
            // 2x2 diagonal block
            else {
                // Permute
                kp = -ipiv[k] + 1;
                if (kp != k + 1) {
                    Blas<T>::swap(&N, &B(k + 1, 0), &ldb, &B(kp, 0), &ldb);
                }

                // Apply inv(L(k))
                local_size = M - k - 2;
                if (local_size >= 0) {
                    Blas<T>::ger(&local_size, &N, &minus_one, &A(k + 2, k), &ione, &B(k, 0), &ldb, &B(k + 2, 0), &ldb);
                    Blas<T>::ger(&local_size, &N, &minus_one, &A(k + 2, k + 1), &ione, &B(k + 1, 0), &ldb, &B(k + 2, 0), &ldb);
                }

                // Inverse of D(k)
                akm1k = A(k + 1, k);
                akm1  = A(k, k) / akm1k;
                ak    = A(k + 1, k + 1) / akm1k;
                denom = akm1 * ak - T(1);

                for (int j = 0; j < N; j++) {
                    bkm1        = B(k, j) / akm1k;
                    bk          = B(k + 1, j) / akm1k;
                    B(k, j)     = (ak * bkm1 - bk) / denom;
                    B(k + 1, j) = (akm1 * bk - bkm1) / denom;
                }

                // Increment
                k += 2;
            }
        }
    } else if (UPLO == 'L' && transa == 'T' && side == 'L') {
        int k = M - 1;
        while (k >= 0) {
            // 1x1 diagonal block
            if (ipiv[k] > 0) {

                local_size = M - k - 1;
                if (local_size >= 0) {
                    Blas<T>::gemv(&transa, &local_size, &N, &minus_one, &B(k + 1, 0), &ldb, &A(k + 1, k), &ione, &one, &B(k, 0), &ldb);
                }

                kp = ipiv[k] - 1;
                if (kp != k) {
                    Blas<T>::swap(&N, &B(k, 0), &ldb, &B(kp, 0), &ldb);
                }

                // Increment
                k -= 1;
            }
            // 2x2 diagonal block
            else {

                local_size = M - k - 1;
                if (local_size >= 0) {
                    Blas<T>::gemv(&transa, &local_size, &N, &minus_one, &B(k + 1, 0), &ldb, &A(k + 1, k), &ione, &one, &B(k, 0), &ldb);
                    Blas<T>::gemv(&transa, &local_size, &N, &minus_one, &B(k + 1, 0), &ldb, &A(k + 1, k - 1), &ione, &one, &B(k - 1, 0), &ldb);
                }

                kp = -ipiv[k] + 1;
                if (kp != k) {
                    Blas<T>::swap(&N, &B(k, 0), &ldb, &B(kp, 0), &ldb);
                }

                // Increment
                k -= 2;
            }
        }
    } else if (UPLO == 'U' && transa == 'N' && side == 'L') {
        int k = M - 1;
        while (k >= 0) {
            // 1x1 diagonal block
            if (ipiv[k] > 0) {
                // Permute
                kp = ipiv[k] - 1;
                if (kp != k) {
                    Blas<T>::swap(&N, &B(k, 0), &ldb, &B(kp, 0), &ldb);
                }

                // Apply inv(L(k))
                local_size = k;
                if (local_size >= 0) {
                    Blas<T>::ger(&local_size, &N, &minus_one, &A(0, k), &ione, &B(k, 0), &ldb, &B(0, 0), &ldb);
                }

                // Inverse of D(k)
                factor = 1. / A(k, k);
                Blas<T>::scal(&N, &factor, &B(k, 0), &ldb);

                // Increment
                k -= 1;
            }
            // 2x2 diagonal block
            else {
                // Permute
                kp = -ipiv[k] + 1;
                if (kp != k - 1) {
                    Blas<T>::swap(&N, &B(k - 1, 0), &ldb, &B(kp, 0), &ldb);
                }

                // Apply inv(L(k))
                local_size = k - 1;
                if (local_size >= 0) {
                    Blas<T>::ger(&local_size, &N, &minus_one, &A(0, k), &ione, &B(k, 0), &ldb, &B(0, 0), &ldb);
                    Blas<T>::ger(&local_size, &N, &minus_one, &A(0, k - 1), &ione, &B(k - 1, 0), &ldb, &B(0, 0), &ldb);
                }
                // Inverse of D(k)
                akm1k = A(k - 1, k);
                akm1  = A(k - 1, k - 1) / akm1k;
                ak    = A(k, k) / akm1k;
                denom = akm1 * ak - T(1);

                for (int j = 0; j < N; j++) {
                    bkm1        = B(k - 1, j) / akm1k;
                    bk          = B(k, j) / akm1k;
                    B(k - 1, j) = (ak * bkm1 - bk) / denom;
                    B(k, j)     = (akm1 * bk - bkm1) / denom;
                }

                // Increment
                k -= 2;
            }
        }
    } else if (UPLO == 'U' && transa == 'T' && side == 'L') {
        int k = 0;
        while (k < M) {
            // 1x1 diagonal block
            if (ipiv[k] > 0) {

                local_size = k;
                if (local_size >= 0) {
                    Blas<T>::gemv(&transa, &local_size, &N, &minus_one, &B(0, 0), &ldb, &A(0, k), &ione, &one, &B(k, 0), &ldb);
                }

                kp = ipiv[k] - 1;
                if (kp != k) {
                    Blas<T>::swap(&N, &B(k, 0), &ldb, &B(kp, 0), &ldb);
                }

                // Increment
                k += 1;
            }
            // 2x2 diagonal block
            else {

                local_size = k;
                if (local_size >= 0) {
                    Blas<T>::gemv(&transa, &local_size, &N, &minus_one, &B(0, 0), &ldb, &A(0, k), &ione, &one, &B(k, 0), &ldb);
                    Blas<T>::gemv(&transa, &local_size, &N, &minus_one, &B(0, 0), &ldb, &A(0, k + 1), &ione, &one, &B(k + 1, 0), &ldb);
                }

                kp = -ipiv[k] + 1;
                if (kp != k) {
                    Blas<T>::swap(&N, &B(k, 0), &ldb, &B(kp, 0), &ldb);
                }

                // Increment
                k += 2;
            }
        }
    } else if (UPLO == 'L' && transa == 'T' && side == 'R') {
        int k = 0;
        while (k < M) {
            std::cout << k << "\n";
            // 1x1 diagonal block
            if (ipiv[k] > 0) {
                // Permute
                kp = ipiv[k] - 1;
                if (kp != k) {
                    Blas<T>::swap(&N, &B(0, k), &ione, &B(0, kp), &ione);
                }

                // Apply inv(L(k))
                local_size = M - k - 1;
                if (local_size >= 0) {
                    Blas<T>::ger(&N, &local_size, &minus_one, &B(0, k), &ione, &A(k + 1, k), &ione, &B(0, k + 1), &ldb);
                }

                // Inverse of D(k)
                factor = 1. / A(k, k);
                Blas<T>::scal(&N, &factor, &B(0, k), &ione);

                // Increment
                k += 1;
            }
            // 2x2 diagonal block
            else {
                // Permute
                kp = -ipiv[k] + 1;
                if (kp != k + 1) {
                    Blas<T>::swap(&N, &B(0, k + 1), &ione, &B(0, kp), &ione);
                }

                // Apply inv(L(k))
                local_size = M - k - 2;
                if (local_size >= 0) {
                    Blas<T>::ger(&local_size, &N, &minus_one, &B(0, k), &ione, &A(k + 2, k), &ione, &B(0, k + 2), &ldb);
                    Blas<T>::ger(&local_size, &N, &minus_one, &B(0, k + 1), &ione, &A(k + 2, k + 1), &ione, &B(0, k + 2), &ldb);
                }

                // Inverse of D(k)
                akm1k = A(k + 1, k);
                akm1  = A(k, k) / akm1k;
                ak    = A(k + 1, k + 1) / akm1k;
                denom = akm1 * ak - T(1);

                for (int j = 0; j < N; j++) {
                    bkm1        = B(j, k) / akm1k;
                    bk          = B(j, k + 1) / akm1k;
                    B(j, k)     = (ak * bkm1 - bk) / denom;
                    B(j, k + 1) = (akm1 * bk - bkm1) / denom;
                }

                // Increment
                k += 2;
            }
        }
    } else if (UPLO == 'L' && transa == 'N' && side == 'R') {
        int k = M - 1;
        while (k >= 0) {
            // 1x1 diagonal block
            if (ipiv[k] > 0) {

                local_size = M - k - 1;
                if (local_size >= 0) {
                    Blas<T>::gemv(&transa, &N, &local_size, &minus_one, &B(0, k + 1), &ldb, &A(k + 1, k), &ione, &one, &B(0, k), &ldb);
                }

                kp = ipiv[k] - 1;
                if (kp != k) {
                    Blas<T>::swap(&N, &B(0, k), &ione, &B(0, kp), &ione);
                }

                // Increment
                k -= 1;
            }
            // 2x2 diagonal block
            else {

                local_size = M - k - 1;
                if (local_size >= 0) {
                    Blas<T>::gemv(&transa, &N, &local_size, &minus_one, &B(0, k + 1), &ldb, &A(k + 1, k), &ione, &one, &B(0, k), &ldb);
                    Blas<T>::gemv(&transa, &N, &local_size, &minus_one, &B(0, k + 1), &ldb, &A(k + 1, k - 1), &ione, &one, &B(0, k - 1), &ldb);
                }

                kp = -ipiv[k] + 1;
                if (kp != k) {
                    Blas<T>::swap(&N, &B(0, k), &ione, &B(0, kp), &ione);
                }

                // Increment
                k -= 2;
            }
        }
    } else {
        std::cout << "not supported\n";
        exit(1);
    }
}

template <typename T>
void symmetric_ldlt_solve(char UPLO, const Matrix<T> &A, Matrix<T> &B) {
    int M      = A.nb_rows();
    int NRHS   = B.nb_cols();
    int lda    = M;
    int ldb    = M;
    auto &ipiv = A.get_pivots();
    int info;

    Lapack<T>::sytrs(&UPLO, &M, &NRHS, A.data(), &lda, ipiv.data(), B.data(), &ldb, &info);
}

template <typename T>
void hermitian_ldlt_factorization(char UPLO, Matrix<T> &A) {
    int N      = A.nb_rows();
    int lda    = N;
    auto &ipiv = A.get_pivots();
    ipiv.resize(N);
    std::vector<T> work(1);
    int lwork = -1;
    int info;

    Lapack<T>::hetrf(&UPLO, &N, A.data(), &lda, ipiv.data(), work.data(), &lwork, &info);
    lwork = static_cast<int>(std::real(work[0]));
    work.resize(lwork);
    Lapack<T>::hetrf(&UPLO, &N, A.data(), &lda, ipiv.data(), work.data(), &lwork, &info);
}

template <typename T>
void hermitian_ldlt_solve(char UPLO, const Matrix<T> &A, Matrix<T> &B) {
    int M      = A.nb_rows();
    int NRHS   = B.nb_cols();
    int lda    = M;
    int ldb    = M;
    auto &ipiv = A.get_pivots();
    int info;

    Lapack<T>::hetrs(&UPLO, &M, &NRHS, A.data(), &lda, ipiv.data(), B.data(), &ldb, &info);
}

} // namespace htool
#endif
