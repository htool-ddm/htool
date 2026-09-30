#ifndef HTOOL_MATRIX_LINALG_FACTORIZATION_HPP
#define HTOOL_MATRIX_LINALG_FACTORIZATION_HPP

#include "../../matrix/matrix.hpp"           // for Matrix
#include "../../misc/logger.hpp"             // for Logger
#include "../../misc/misc.hpp"               // for is_complex
#include "../../wrappers/wrapper_blas.hpp"   // for Blas
#include "../../wrappers/wrapper_lapack.hpp" // for Lapack
#include <algorithm>                         // for min
#include <string>                            // for to_string

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
    if (info > 0) {
        htool::Logger::get_instance().log(LogLevel::ERROR, "cholesky_factorization failed: the leading minor of order " + std::to_string(info) + " is not positive definite. An indefinite matrix needs an LDLt factorization."); // LCOV_EXCL_LINE
    }
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

// sytrf/hetrf and sytrs/hetrs: LAPACK only has hetrf/hetrs for complex T, and for real T a Hermitian
// LDLt is a symmetric one. Overloaded on is_complex_t to stay C++14-compatible.
template <typename T, typename std::enable_if<!is_complex_t<T>::value, int>::type = 0>
void ldlt_trf(char, const char *UPLO, const int *N, T *A, const int *lda, int *ipiv, T *work, int *lwork, int *info) {
    Lapack<T>::sytrf(UPLO, N, A, lda, ipiv, work, lwork, info);
}

template <typename T, typename std::enable_if<is_complex_t<T>::value, int>::type = 0>
void ldlt_trf(char symmetry, const char *UPLO, const int *N, T *A, const int *lda, int *ipiv, T *work, int *lwork, int *info) {
    if (symmetry == 'H') {
        Lapack<T>::hetrf(UPLO, N, A, lda, ipiv, work, lwork, info);
    } else {
        Lapack<T>::sytrf(UPLO, N, A, lda, ipiv, work, lwork, info);
    }
}

template <typename T, typename std::enable_if<!is_complex_t<T>::value, int>::type = 0>
void ldlt_trs(char, const char *UPLO, const int *N, const int *NRHS, const T *A, const int *lda, const int *ipiv, T *B, const int *ldb, int *info) {
    Lapack<T>::sytrs(UPLO, N, NRHS, A, lda, ipiv, B, ldb, info);
}

template <typename T, typename std::enable_if<is_complex_t<T>::value, int>::type = 0>
void ldlt_trs(char symmetry, const char *UPLO, const int *N, const int *NRHS, const T *A, const int *lda, const int *ipiv, T *B, const int *ldb, int *info) {
    if (symmetry == 'H') {
        Lapack<T>::hetrs(UPLO, N, NRHS, A, lda, ipiv, B, ldb, info);
    } else {
        Lapack<T>::sytrs(UPLO, N, NRHS, A, lda, ipiv, B, ldb, info);
    }
}

// Bunch-Kaufman LDLt: sytrf for symmetry='S', hetrf for symmetry='H' (sytrf for real T, where the two coincide).
template <typename T>
void ldlt_factorization(char symmetry, char UPLO, Matrix<T> &A) {
    int N      = A.nb_rows();
    int lda    = N;
    auto &ipiv = A.get_pivots();
    ipiv.resize(N);
    std::vector<T> work(1);
    int lwork = -1;
    int info;

    ldlt_trf(symmetry, &UPLO, &N, A.data(), &lda, ipiv.data(), work.data(), &lwork, &info);
    lwork = static_cast<int>(std::real(work[0]));
    work.resize(lwork);
    ldlt_trf(symmetry, &UPLO, &N, A.data(), &lda, ipiv.data(), work.data(), &lwork, &info);
    if (info > 0) {
        htool::Logger::get_instance().log(LogLevel::ERROR, "ldlt_factorization: D(" + std::to_string(info) + "," + std::to_string(info) + ") is exactly zero, the matrix is singular."); // LCOV_EXCL_LINE
    }
}

// BLAS has no plain complex ger; geru (unconjugated) is what a complex-symmetric solve needs.
template <typename T, typename std::enable_if<!is_complex_t<T>::value, int>::type = 0>
void ldlt_ger(const int *M, const int *N, const T *alpha, const T *x, const int *incx, const T *y, const int *incy, T *A, const int *lda) {
    Blas<T>::ger(M, N, alpha, x, incx, y, incy, A, lda);
}

template <typename T, typename std::enable_if<is_complex_t<T>::value, int>::type = 0>
void ldlt_ger(const int *M, const int *N, const T *alpha, const T *x, const int *incx, const T *y, const int *incy, T *A, const int *lda) {
    Blas<T>::geru(M, N, alpha, x, incx, y, incy, A, lda);
}

// gerc (conjugates y) instead of geru: used where the Hermitian solve needs L^-H, not L^-T.
template <typename T, typename std::enable_if<!is_complex_t<T>::value, int>::type = 0>
void ldlt_gerc(const int *M, const int *N, const T *alpha, const T *x, const int *incx, const T *y, const int *incy, T *A, const int *lda) {
    Blas<T>::ger(M, N, alpha, x, incx, y, incy, A, lda);
}

template <typename T, typename std::enable_if<is_complex_t<T>::value, int>::type = 0>
void ldlt_gerc(const int *M, const int *N, const T *alpha, const T *x, const int *incx, const T *y, const int *incy, T *A, const int *lda) {
    Blas<T>::gerc(M, N, alpha, x, incx, y, incy, A, lda);
}

// Solves with the unit L/U of an LDLt factorization (Bunch-Kaufman permutations included), transa
// in {N,T,C} like trsm; D is applied separately with apply_ldlt_diagonal. A full solve is transa='N',
// D^-1, then transa='T' (symmetric factorization, as sytrs) or 'C' (Hermitian, as hetrs).
template <typename T>
void triangular_ldlt_matrix_matrix_solve(char side, char UPLO, char transa, T alpha, const Matrix<T> &A, Matrix<T> &B) {
    int M      = A.nb_rows();
    int N      = side == 'L' ? B.nb_cols() : B.nb_rows();
    auto &ipiv = A.get_pivots();
    int ldb    = B.nb_rows();
    int local_size;
    int kp;
    int ione    = 1;
    T one       = 1;
    T minus_one = -1;

    if (alpha != T(1)) {
        int b_size = B.nb_rows() * B.nb_cols();
        Blas<T>::scal(&b_size, &alpha, B.data(), &ione);
    }

    if (UPLO == 'L' && transa == 'N' && side == 'L') {
        int k = 0;
        while (k < M) {
            if (ipiv[k] > 0) {
                kp = ipiv[k] - 1;
                if (kp != k) {
                    Blas<T>::swap(&N, &B(k, 0), &ldb, &B(kp, 0), &ldb);
                }

                local_size = M - k - 1;
                if (local_size >= 0) {
                    ldlt_ger(&local_size, &N, &minus_one, &A(k + 1, k), &ione, &B(k, 0), &ldb, &B(k + 1, 0), &ldb);
                }

                k += 1;
            } else {
                kp = -ipiv[k] - 1;
                if (kp != k + 1) {
                    Blas<T>::swap(&N, &B(k + 1, 0), &ldb, &B(kp, 0), &ldb);
                }

                local_size = M - k - 2;
                if (local_size >= 0) {
                    ldlt_ger(&local_size, &N, &minus_one, &A(k + 2, k), &ione, &B(k, 0), &ldb, &B(k + 2, 0), &ldb);
                    ldlt_ger(&local_size, &N, &minus_one, &A(k + 2, k + 1), &ione, &B(k + 1, 0), &ldb, &B(k + 2, 0), &ldb);
                }

                k += 2;
            }
        }
    } else if (UPLO == 'L' && transa == 'T' && side == 'L') {
        int k = M - 1;
        while (k >= 0) {
            if (ipiv[k] > 0) {
                local_size = M - k - 1;
                if (local_size >= 0) {
                    Blas<T>::gemv(&transa, &local_size, &N, &minus_one, &B(k + 1, 0), &ldb, &A(k + 1, k), &ione, &one, &B(k, 0), &ldb);
                }

                kp = ipiv[k] - 1;
                if (kp != k) {
                    Blas<T>::swap(&N, &B(k, 0), &ldb, &B(kp, 0), &ldb);
                }
                k -= 1;
            } else {
                local_size = M - k - 1;
                if (local_size >= 0) {
                    Blas<T>::gemv(&transa, &local_size, &N, &minus_one, &B(k + 1, 0), &ldb, &A(k + 1, k), &ione, &one, &B(k, 0), &ldb);
                    Blas<T>::gemv(&transa, &local_size, &N, &minus_one, &B(k + 1, 0), &ldb, &A(k + 1, k - 1), &ione, &one, &B(k - 1, 0), &ldb);
                }

                kp = -ipiv[k] - 1;
                if (kp != k) {
                    Blas<T>::swap(&N, &B(k, 0), &ldb, &B(kp, 0), &ldb);
                }
                k -= 2;
            }
        }
    } else if (UPLO == 'U' && transa == 'N' && side == 'L') {
        int k = M - 1;
        while (k >= 0) {
            if (ipiv[k] > 0) {
                kp = ipiv[k] - 1;
                if (kp != k) {
                    Blas<T>::swap(&N, &B(k, 0), &ldb, &B(kp, 0), &ldb);
                }

                local_size = k;
                if (local_size >= 0) {
                    ldlt_ger(&local_size, &N, &minus_one, &A(0, k), &ione, &B(k, 0), &ldb, &B(0, 0), &ldb);
                }

                k -= 1;
            } else {
                kp = -ipiv[k] - 1;
                if (kp != k - 1) {
                    Blas<T>::swap(&N, &B(k - 1, 0), &ldb, &B(kp, 0), &ldb);
                }

                local_size = k - 1;
                if (local_size >= 0) {
                    ldlt_ger(&local_size, &N, &minus_one, &A(0, k), &ione, &B(k, 0), &ldb, &B(0, 0), &ldb);
                    ldlt_ger(&local_size, &N, &minus_one, &A(0, k - 1), &ione, &B(k - 1, 0), &ldb, &B(0, 0), &ldb);
                }

                k -= 2;
            }
        }
    } else if (UPLO == 'U' && transa == 'T' && side == 'L') {
        int k = 0;
        while (k < M) {
            if (ipiv[k] > 0) {
                local_size = k;
                if (local_size >= 0) {
                    Blas<T>::gemv(&transa, &local_size, &N, &minus_one, &B(0, 0), &ldb, &A(0, k), &ione, &one, &B(k, 0), &ldb);
                }

                kp = ipiv[k] - 1;
                if (kp != k) {
                    Blas<T>::swap(&N, &B(k, 0), &ldb, &B(kp, 0), &ldb);
                }
                k += 1;
            } else {
                local_size = k;
                if (local_size >= 0) {
                    Blas<T>::gemv(&transa, &local_size, &N, &minus_one, &B(0, 0), &ldb, &A(0, k), &ione, &one, &B(k, 0), &ldb);
                    Blas<T>::gemv(&transa, &local_size, &N, &minus_one, &B(0, 0), &ldb, &A(0, k + 1), &ione, &one, &B(k + 1, 0), &ldb);
                }

                kp = -ipiv[k] - 1;
                if (kp != k) {
                    Blas<T>::swap(&N, &B(k, 0), &ldb, &B(kp, 0), &ldb);
                }
                k += 2;
            }
        }
    } else if (UPLO == 'L' && transa == 'T' && side == 'R') {
        int k = 0;
        while (k < M) {
            if (ipiv[k] > 0) {
                kp = ipiv[k] - 1;
                if (kp != k) {
                    Blas<T>::swap(&N, &B(0, k), &ione, &B(0, kp), &ione);
                }

                local_size = M - k - 1;
                if (local_size >= 0) {
                    ldlt_ger(&N, &local_size, &minus_one, &B(0, k), &ione, &A(k + 1, k), &ione, &B(0, k + 1), &ldb);
                }

                k += 1;
            } else {
                kp = -ipiv[k] - 1;
                if (kp != k + 1) {
                    Blas<T>::swap(&N, &B(0, k + 1), &ione, &B(0, kp), &ione);
                }

                local_size = M - k - 2;
                if (local_size >= 0) {
                    ldlt_ger(&N, &local_size, &minus_one, &B(0, k), &ione, &A(k + 2, k), &ione, &B(0, k + 2), &ldb);
                    ldlt_ger(&N, &local_size, &minus_one, &B(0, k + 1), &ione, &A(k + 2, k + 1), &ione, &B(0, k + 2), &ldb);
                }

                k += 2;
            }
        }
    } else if (UPLO == 'L' && transa == 'N' && side == 'R') {
        int k = M - 1;
        while (k >= 0) {
            if (ipiv[k] > 0) {
                local_size = M - k - 1;
                if (local_size >= 0) {
                    Blas<T>::gemv(&transa, &N, &local_size, &minus_one, &B(0, k + 1), &ldb, &A(k + 1, k), &ione, &one, &B(0, k), &ione);
                }

                kp = ipiv[k] - 1;
                if (kp != k) {
                    Blas<T>::swap(&N, &B(0, k), &ione, &B(0, kp), &ione);
                }
                k -= 1;
            } else {
                local_size = M - k - 1;
                if (local_size >= 0) {
                    Blas<T>::gemv(&transa, &N, &local_size, &minus_one, &B(0, k + 1), &ldb, &A(k + 1, k), &ione, &one, &B(0, k), &ione);
                    Blas<T>::gemv(&transa, &N, &local_size, &minus_one, &B(0, k + 1), &ldb, &A(k + 1, k - 1), &ione, &one, &B(0, k - 1), &ione);
                }

                kp = -ipiv[k] - 1;
                if (kp != k) {
                    Blas<T>::swap(&N, &B(0, k), &ione, &B(0, kp), &ione);
                }
                k -= 2;
            }
        }
    } else if (UPLO == 'U' && transa == 'T' && side == 'R') {
        int k = M - 1;
        while (k >= 0) {
            if (ipiv[k] > 0) {
                kp = ipiv[k] - 1;
                if (kp != k) {
                    Blas<T>::swap(&N, &B(0, k), &ione, &B(0, kp), &ione);
                }

                local_size = k;
                if (local_size >= 0) {
                    ldlt_ger(&N, &local_size, &minus_one, &B(0, k), &ione, &A(0, k), &ione, &B(0, 0), &ldb);
                }

                k -= 1;
            } else {
                kp = -ipiv[k] - 1;
                if (kp != k - 1) {
                    Blas<T>::swap(&N, &B(0, k - 1), &ione, &B(0, kp), &ione);
                }

                local_size = k - 1;
                if (local_size >= 0) {
                    ldlt_ger(&N, &local_size, &minus_one, &B(0, k), &ione, &A(0, k), &ione, &B(0, 0), &ldb);
                    ldlt_ger(&N, &local_size, &minus_one, &B(0, k - 1), &ione, &A(0, k - 1), &ione, &B(0, 0), &ldb);
                }

                k -= 2;
            }
        }
    } else if (UPLO == 'U' && transa == 'N' && side == 'R') {
        int k = 0;
        while (k < M) {
            if (ipiv[k] > 0) {
                local_size = k;
                if (local_size >= 0) {
                    Blas<T>::gemv(&transa, &N, &local_size, &minus_one, &B(0, 0), &ldb, &A(0, k), &ione, &one, &B(0, k), &ione);
                }

                kp = ipiv[k] - 1;
                if (kp != k) {
                    Blas<T>::swap(&N, &B(0, k), &ione, &B(0, kp), &ione);
                }
                k += 1;
            } else {
                local_size = k;
                if (local_size >= 0) {
                    Blas<T>::gemv(&transa, &N, &local_size, &minus_one, &B(0, 0), &ldb, &A(0, k), &ione, &one, &B(0, k), &ione);
                    Blas<T>::gemv(&transa, &N, &local_size, &minus_one, &B(0, 0), &ldb, &A(0, k + 1), &ione, &one, &B(0, k + 1), &ione);
                }

                kp = -ipiv[k] - 1;
                if (kp != k) {
                    Blas<T>::swap(&N, &B(0, k), &ione, &B(0, kp), &ione);
                }
                k += 2;
            }
        }
    } else if (UPLO == 'L' && transa == 'C' && side == 'L') {
        int k = M - 1;
        while (k >= 0) {
            if (ipiv[k] > 0) {
                local_size = M - k - 1;
                if (local_size >= 0) {
                    for (int j = 0; j < N; j++) {
                        B(k, j) = conj_if_complex(B(k, j));
                    }
                    Blas<T>::gemv(&transa, &local_size, &N, &minus_one, &B(k + 1, 0), &ldb, &A(k + 1, k), &ione, &one, &B(k, 0), &ldb);
                    for (int j = 0; j < N; j++) {
                        B(k, j) = conj_if_complex(B(k, j));
                    }
                }

                kp = ipiv[k] - 1;
                if (kp != k) {
                    Blas<T>::swap(&N, &B(k, 0), &ldb, &B(kp, 0), &ldb);
                }
                k -= 1;
            } else {
                local_size = M - k - 1;
                if (local_size >= 0) {
                    for (int j = 0; j < N; j++) {
                        B(k, j) = conj_if_complex(B(k, j));
                    }
                    Blas<T>::gemv(&transa, &local_size, &N, &minus_one, &B(k + 1, 0), &ldb, &A(k + 1, k), &ione, &one, &B(k, 0), &ldb);
                    for (int j = 0; j < N; j++) {
                        B(k, j) = conj_if_complex(B(k, j));
                    }

                    for (int j = 0; j < N; j++) {
                        B(k - 1, j) = conj_if_complex(B(k - 1, j));
                    }
                    Blas<T>::gemv(&transa, &local_size, &N, &minus_one, &B(k + 1, 0), &ldb, &A(k + 1, k - 1), &ione, &one, &B(k - 1, 0), &ldb);
                    for (int j = 0; j < N; j++) {
                        B(k - 1, j) = conj_if_complex(B(k - 1, j));
                    }
                }

                kp = -ipiv[k] - 1;
                if (kp != k) {
                    Blas<T>::swap(&N, &B(k, 0), &ldb, &B(kp, 0), &ldb);
                }
                k -= 2;
            }
        }
    } else if (UPLO == 'U' && transa == 'C' && side == 'L') {
        int k = 0;
        while (k < M) {
            if (ipiv[k] > 0) {
                local_size = k;
                if (local_size >= 0) {
                    for (int j = 0; j < N; j++) {
                        B(k, j) = conj_if_complex(B(k, j));
                    }
                    Blas<T>::gemv(&transa, &local_size, &N, &minus_one, &B(0, 0), &ldb, &A(0, k), &ione, &one, &B(k, 0), &ldb);
                    for (int j = 0; j < N; j++) {
                        B(k, j) = conj_if_complex(B(k, j));
                    }
                }

                kp = ipiv[k] - 1;
                if (kp != k) {
                    Blas<T>::swap(&N, &B(k, 0), &ldb, &B(kp, 0), &ldb);
                }
                k += 1;
            } else {
                local_size = k;
                if (local_size >= 0) {
                    for (int j = 0; j < N; j++) {
                        B(k, j) = conj_if_complex(B(k, j));
                    }
                    Blas<T>::gemv(&transa, &local_size, &N, &minus_one, &B(0, 0), &ldb, &A(0, k), &ione, &one, &B(k, 0), &ldb);
                    for (int j = 0; j < N; j++) {
                        B(k, j) = conj_if_complex(B(k, j));
                    }

                    for (int j = 0; j < N; j++) {
                        B(k + 1, j) = conj_if_complex(B(k + 1, j));
                    }
                    Blas<T>::gemv(&transa, &local_size, &N, &minus_one, &B(0, 0), &ldb, &A(0, k + 1), &ione, &one, &B(k + 1, 0), &ldb);
                    for (int j = 0; j < N; j++) {
                        B(k + 1, j) = conj_if_complex(B(k + 1, j));
                    }
                }

                kp = -ipiv[k] - 1;
                if (kp != k) {
                    Blas<T>::swap(&N, &B(k, 0), &ldb, &B(kp, 0), &ldb);
                }
                k += 2;
            }
        }
    } else if (UPLO == 'L' && transa == 'C' && side == 'R') {
        int k = 0;
        while (k < M) {
            if (ipiv[k] > 0) {
                kp = ipiv[k] - 1;
                if (kp != k) {
                    Blas<T>::swap(&N, &B(0, k), &ione, &B(0, kp), &ione);
                }

                local_size = M - k - 1;
                if (local_size >= 0) {
                    ldlt_gerc(&N, &local_size, &minus_one, &B(0, k), &ione, &A(k + 1, k), &ione, &B(0, k + 1), &ldb);
                }

                k += 1;
            } else {
                kp = -ipiv[k] - 1;
                if (kp != k + 1) {
                    Blas<T>::swap(&N, &B(0, k + 1), &ione, &B(0, kp), &ione);
                }

                local_size = M - k - 2;
                if (local_size >= 0) {
                    ldlt_gerc(&N, &local_size, &minus_one, &B(0, k), &ione, &A(k + 2, k), &ione, &B(0, k + 2), &ldb);
                    ldlt_gerc(&N, &local_size, &minus_one, &B(0, k + 1), &ione, &A(k + 2, k + 1), &ione, &B(0, k + 2), &ldb);
                }

                k += 2;
            }
        }
    } else if (UPLO == 'U' && transa == 'C' && side == 'R') {
        int k = M - 1;
        while (k >= 0) {
            if (ipiv[k] > 0) {
                kp = ipiv[k] - 1;
                if (kp != k) {
                    Blas<T>::swap(&N, &B(0, k), &ione, &B(0, kp), &ione);
                }

                local_size = k;
                if (local_size >= 0) {
                    ldlt_gerc(&N, &local_size, &minus_one, &B(0, k), &ione, &A(0, k), &ione, &B(0, 0), &ldb);
                }

                k -= 1;
            } else {
                kp = -ipiv[k] - 1;
                if (kp != k - 1) {
                    Blas<T>::swap(&N, &B(0, k - 1), &ione, &B(0, kp), &ione);
                }

                local_size = k - 1;
                if (local_size >= 0) {
                    ldlt_gerc(&N, &local_size, &minus_one, &B(0, k), &ione, &A(0, k), &ione, &B(0, 0), &ldb);
                    ldlt_gerc(&N, &local_size, &minus_one, &B(0, k - 1), &ione, &A(0, k - 1), &ione, &B(0, 0), &ldb);
                }

                k -= 2;
            }
        }
    } else {
        std::cout << "not supported\n"; // LCOV_EXCL_LINE
        exit(1);                        // LCOV_EXCL_LINE
    }
}

// Applies D^-1, D being the block-diagonal (1x1/2x2) factor of an LDLt factorization:
// B := D^-1 * B for side='L', B := B * D^-1 for side='R'. The 2x2 inverse uses sytrs/hetrs's scaled
// form (dividing by the off-diagonal entries first) instead of a determinant.
template <typename T>
void apply_ldlt_diagonal(char symmetry, char side, char UPLO, const Matrix<T> &A, Matrix<T> &B) {
    bool hermitian = symmetry == 'H';
    int M          = A.nb_rows();
    auto &ipiv     = A.get_pivots();
    // Row/column k of B is B.data()[k * k_stride + j * j_stride], j running over the other dimension.
    int N        = side == 'L' ? B.nb_cols() : B.nb_rows();
    int k_stride = side == 'L' ? 1 : B.nb_rows();
    int j_stride = side == 'L' ? B.nb_rows() : 1;
    T *b         = B.data();

    int k = 0;
    while (k < M) {
        if (ipiv[k] > 0) {
            T d_inverse = T(1) / (hermitian ? T(std::real(A(k, k))) : A(k, k));
            for (int j = 0; j < N; j++) {
                b[k * k_stride + j * j_stride] *= d_inverse;
            }
            k += 1;
        } else {
            T d11 = hermitian ? T(std::real(A(k, k))) : A(k, k);
            T d22 = hermitian ? T(std::real(A(k + 1, k + 1))) : A(k + 1, k + 1);
            T d21 = UPLO == 'L' ? A(k + 1, k) : (hermitian ? conj_if_complex(A(k, k + 1)) : A(k, k + 1));
            T d12 = hermitian ? conj_if_complex(d21) : d21;
            if (side == 'R') { // x D = y  <=>  D^T x^T = y^T
                std::swap(d12, d21);
            }
            T akm1 = d11 / d12, ak = d22 / d21, denom = akm1 * ak - T(1);
            for (int j = 0; j < N; j++) {
                T &b1  = b[k * k_stride + j * j_stride];
                T &b2  = b[(k + 1) * k_stride + j * j_stride];
                T bkm1 = b1 / d12;
                T bk   = b2 / d21;
                b1     = (ak * bkm1 - bk) / denom;
                b2     = (akm1 * bk - bkm1) / denom;
            }
            k += 2;
        }
    }
}

// Solves with an ldlt_factorization of the same symmetry: sytrs for 'S', hetrs for 'H' (sytrs for real T).
template <typename T>
void ldlt_solve(char symmetry, char UPLO, const Matrix<T> &A, Matrix<T> &B) {
    int M      = A.nb_rows();
    int NRHS   = B.nb_cols();
    int lda    = M;
    int ldb    = M;
    auto &ipiv = A.get_pivots();
    int info;

    ldlt_trs(symmetry, &UPLO, &M, &NRHS, A.data(), &lda, ipiv.data(), B.data(), &ldb, &info);
}

} // namespace htool
#endif
