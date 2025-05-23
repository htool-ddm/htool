#ifndef HTOOL_BLAS_HPP
#define HTOOL_BLAS_HPP

#include "../misc/define.hpp"
#include "../misc/misc.hpp"
#include <complex>

#if defined(__powerpc__) || defined(INTEL_MKL_VERSION)
#    define HTOOL_BLAS_F77(func) func
#else
#    define HTOOL_BLAS_F77(func) func##_
#endif

#if HTOOL_MKL
#    if defined(__cplusplus) && (!defined(INTEL_MKL_VERSION) || INTEL_MKL_VERSION >= 20200004)
#        define HTOOL_NOEXCEPT noexcept
#    endif
#endif

#if !HTOOL_MKL || !defined(__cplusplus) || (defined(INTEL_MKL_VERSION) && INTEL_MKL_VERSION < 20200004)
#    define HTOOL_NOEXCEPT
#endif

#define HTOOL_GENERATE_EXTERN_BLAS(C, T)                                                                                                                                                                                 \
    void HTOOL_BLAS_F77(C##axpy)(const int *, const T *, const T *, const int *, T *, const int *) HTOOL_NOEXCEPT;                                                                                                       \
    void HTOOL_BLAS_F77(C##scal)(const int *, const T *, T *, const int *) HTOOL_NOEXCEPT;                                                                                                                               \
    void HTOOL_BLAS_F77(C##gemv)(const char *, const int *, const int *, const T *, const T *, const int *, const T *, const int *, const T *, T *, const int *) HTOOL_NOEXCEPT;                                         \
    void HTOOL_BLAS_F77(C##gemm)(const char *, const char *, const int *, const int *, const int *, const T *, const T *, const int *, const T *, const int *, const T *, T *, const int *) HTOOL_NOEXCEPT;              \
    void HTOOL_BLAS_F77(C##symv)(const char *, const int *, const T *, const T *, const int *, const T *, const int *, const T *, T *, const int *) HTOOL_NOEXCEPT;                                                      \
    void HTOOL_BLAS_F77(C##symm)(const char *, const char *, const int *, const int *, const T *, const T *, const int *, const T *, const int *, const T *, T *, const int *) HTOOL_NOEXCEPT;                           \
    void HTOOL_BLAS_F77(C##syrk)(const char *const, const char *const, const int *const, const int *const, const T *const, const T *const, const int *const, const T *const, T *const, const int *const) HTOOL_NOEXCEPT; \
    void HTOOL_BLAS_F77(C##trsm)(const char *, const char *, const char *, const char *, const int *, const int *, const T *, const T *, const int *, T *, const int *) HTOOL_NOEXCEPT;                                  \
    void HTOOL_BLAS_F77(C##laswp)(const int *, T *, const int *, const int *, const int *, const int *, const int *) HTOOL_NOEXCEPT;
#define HTOOL_GENERATE_EXTERN_BLAS_COMPLEX(C, T, B, U)                                                                                                                                                                   \
    HTOOL_GENERATE_EXTERN_BLAS(B, U)                                                                                                                                                                                     \
    HTOOL_GENERATE_EXTERN_BLAS(C, T)                                                                                                                                                                                     \
    void HTOOL_BLAS_F77(C##hemv)(const char *, const int *, const T *, const T *, const int *, const T *, const int *, const T *, T *, const int *) HTOOL_NOEXCEPT;                                                      \
    void HTOOL_BLAS_F77(C##hemm)(const char *, const char *, const int *, const int *, const T *, const T *, const int *, const T *, const int *, const T *, T *, const int *) HTOOL_NOEXCEPT;                           \
    void HTOOL_BLAS_F77(C##herk)(const char *const, const char *const, const int *const, const int *const, const U *const, const T *const, const int *const, const U *const, T *const, const int *const) HTOOL_NOEXCEPT; \
    U HTOOL_BLAS_F77(B##nrm2)(const int *, const U *, const int *) HTOOL_NOEXCEPT;                                                                                                                                       \
    U HTOOL_BLAS_F77(B##C##nrm2)(const int *, const T *, const int *) HTOOL_NOEXCEPT;

#if HTOOL_MKL
#    define HTOOL_GENERATE_EXTERN_GEMM3M(C, T) \
        void HTOOL_BLAS_F77(C##gemm3m)(const char *, const char *, const int *, const int *, const int *, const T *, const T *, const int *, const T *, const int *, const T *, T *, const int *) HTOOL_NOEXCEPT;
#    define HTOOL_GENERATE_EXTERN_MKL_EXTENSIONS(C, T, B, U) \
        HTOOL_GENERATE_EXTERN_GEMM3M(C, T)
#endif

#ifdef __cplusplus
extern "C" {
HTOOL_GENERATE_EXTERN_BLAS_COMPLEX(c, std::complex<float>, s, float)
HTOOL_GENERATE_EXTERN_BLAS_COMPLEX(z, std::complex<double>, d, double)
#    if HTOOL_MKL
HTOOL_GENERATE_EXTERN_MKL_EXTENSIONS(c, std::complex<float>, s, float)
HTOOL_GENERATE_EXTERN_MKL_EXTENSIONS(z, std::complex<double>, d, double)
#    endif
}
#else
HTOOL_GENERATE_EXTERN_BLAS_COMPLEX(c, void, s, float)
HTOOL_GENERATE_EXTERN_BLAS_COMPLEX(z, void, d, double)
#endif // __cplusplus

#ifdef __cplusplus
namespace htool {
/* Class: Blas
 *
 *  A class that wraps most of BLAS routines for dense linear algebra.
 *
 * Template Parameter:
 *    K              - Scalar type. */
template <class K>
struct Blas {
    /* Function: axpy
     *  Computes a scalar-vector product and adds the result to a vector. */
    static void axpy(const int *const, const K *const, const K *const, const int *const, K *const, const int *const);
    /* Function: scal
     *  Computes the product of a vector by a scalar. */
    static void scal(const int *const, const K *const, K *const, const int *const);
    /* Function: nrm2
     *  Computes the Euclidean norm of a vector. */
    static underlying_type<K> nrm2(const int *const, const K *const, const int *const);
    /* Function: dot
     *  Computes a vector-vector dot product. */
    static K dot(const int *const, const K *const, const int *const, const K *const, const int *const);
    /* Function: gemv
     *  Computes a matrix-vector product. */
    static void gemv(const char *const, const int *const, const int *const, const K *const, const K *const, const int *const, const K *const, const int *const, const K *const, K *const, const int *const);
    /* Function: gemm
     *  Computes a scalar-matrix-matrix product. */
    static void gemm(const char *const, const char *const, const int *const, const int *const, const int *const, const K *const, const K *const, const int *const, const K *const, const int *const, const K *const, K *const, const int *const);
    /* Function: symv
     *  Computes a symmetric matrix-vector product. */
    static void symv(const char *const, const int *const, const K *const, const K *const, const int *const, const K *const, const int *const, const K *const, K *const, const int *const);
    /* Function: symm
     *  Computes a symmetric scalar-matrix-matrix product. */
    static void symm(const char *const, const char *const, const int *const, const int *const, const K *const, const K *const, const int *const, const K *const, const int *const, const K *const, K *const, const int *const);
    /* Function: chemv
     *  Computes a hermitian matrix-vector product. */
    static void hemv(const char *const, const int *const, const K *const, const K *const, const int *const, const K *const, const int *const, const K *const, K *const, const int *const);
    /* Function: chemm
     *  Computes a hermitian scalar-matrix-matrix product. */
    static void hemm(const char *const, const char *const, const int *const, const int *const, const K *const, const K *const, const int *const, const K *const, const int *const, const K *const, K *const, const int *const);
    /* Function: syrk
     *  Computes a symmetric rank-k update. */
    static void syrk(const char *const, const char *const, const int *const, const int *const, const K *const, const K *const, const int *const, const K *const, K *const, const int *const);
    /* Function: herk
     *  Computes a Hermitian rank-k update. */
    static void herk(const char *const, const char *const, const int *const, const int *const, const underlying_type<K> *const, const K *const, const int *const, const underlying_type<K> *const, K *const, const int *const);
    /* Function: trsm
     *  Solves a triangular system. */
    static void trsm(const char *, const char *, const char *, const char *, const int *, const int *, const K *, const K *, const int *, K *, const int *);
    /* Function:
     *  Performs a series of row interchanges on the matrix A */
    static void laswp(const int *, K *, const int *, const int *, const int *, const int *, const int *);
};

#    define HTOOL_GENERATE_GEMM(C, T)                                                                                                                                                                                                                                                                            \
        template <>                                                                                                                                                                                                                                                                                              \
        inline void Blas<T>::gemm(const char *const transa, const char *const transb, const int *const m, const int *const n, const int *const k, const T *const alpha, const T *const a, const int *const lda, const T *const b, const int *const ldb, const T *const beta, T *const c, const int *const ldc) { \
            HTOOL_BLAS_F77(C##gemm)                                                                                                                                                                                                                                                                              \
            (transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);                                                                                                                                                                                                                                      \
        }

#    if !HTOOL_MKL
#        define HTOOL_GENERATE_GEMM_COMPLEX(C, T) HTOOL_GENERATE_GEMM(C, T)
#    else
#        define HTOOL_GENERATE_GEMM_COMPLEX(C, T)                                                                                                                                                                                                                                                                    \
            template <>                                                                                                                                                                                                                                                                                              \
            inline void Blas<T>::gemm(const char *const transa, const char *const transb, const int *const m, const int *const n, const int *const k, const T *const alpha, const T *const a, const int *const lda, const T *const b, const int *const ldb, const T *const beta, T *const c, const int *const ldc) { \
                HTOOL_BLAS_F77(C##gemm3m)                                                                                                                                                                                                                                                                            \
                (transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);                                                                                                                                                                                                                                      \
            }
#    endif

#    define HTOOL_GENERATE_BLAS(C, T)                                                                                                                                                                                                                                                    \
        template <>                                                                                                                                                                                                                                                                      \
        inline void Blas<T>::axpy(const int *const n, const T *const a, const T *const x, const int *const incx, T *const y, const int *const incy) {                                                                                                                                    \
            HTOOL_BLAS_F77(C##axpy)                                                                                                                                                                                                                                                      \
            (n, a, x, incx, y, incy);                                                                                                                                                                                                                                                    \
        }                                                                                                                                                                                                                                                                                \
        template <>                                                                                                                                                                                                                                                                      \
        inline void Blas<T>::scal(const int *const n, const T *const a, T *const x, const int *const incx) {                                                                                                                                                                             \
            HTOOL_BLAS_F77(C##scal)                                                                                                                                                                                                                                                      \
            (n, a, x, incx);                                                                                                                                                                                                                                                             \
        }                                                                                                                                                                                                                                                                                \
        template <>                                                                                                                                                                                                                                                                      \
        inline T Blas<T>::dot(const int *const n, const T *const x, const int *const incx, const T *const y, const int *const incy) {                                                                                                                                                    \
            T sum = T();                                                                                                                                                                                                                                                                 \
            for (int i = 0, j = 0, k = 0; i < *n; ++i, j += *incx, k += *incy)                                                                                                                                                                                                           \
                sum += conj_if_complex(x[j]) * y[k];                                                                                                                                                                                                                                     \
            return sum;                                                                                                                                                                                                                                                                  \
        }                                                                                                                                                                                                                                                                                \
        template <>                                                                                                                                                                                                                                                                      \
        inline void Blas<T>::gemv(const char *const trans, const int *const m, const int *const n, const T *const alpha, const T *const a, const int *const lda, const T *const b, const int *const ldb, const T *const beta, T *const c, const int *const ldc) {                        \
            HTOOL_BLAS_F77(C##gemv)                                                                                                                                                                                                                                                      \
            (trans, m, n, alpha, a, lda, b, ldb, beta, c, ldc);                                                                                                                                                                                                                          \
        }                                                                                                                                                                                                                                                                                \
        template <>                                                                                                                                                                                                                                                                      \
        inline void Blas<T>::symv(const char *const uplo, const int *const n, const T *const alpha, const T *const a, const int *const lda, const T *const x, const int *const incx, const T *const beta, T *const y, const int *const incy) {                                           \
            HTOOL_BLAS_F77(C##symv)                                                                                                                                                                                                                                                      \
            (uplo, n, alpha, a, lda, x, incx, beta, y, incy);                                                                                                                                                                                                                            \
        }                                                                                                                                                                                                                                                                                \
        template <>                                                                                                                                                                                                                                                                      \
        inline void Blas<T>::symm(const char *const side, const char *const uplo, const int *const m, const int *const n, const T *const alpha, const T *const a, const int *const lda, const T *const b, const int *const ldb, const T *const beta, T *const c, const int *const ldc) { \
            HTOOL_BLAS_F77(C##symm)                                                                                                                                                                                                                                                      \
            (side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);                                                                                                                                                                                                                     \
        }                                                                                                                                                                                                                                                                                \
        template <>                                                                                                                                                                                                                                                                      \
        inline void Blas<T>::syrk(const char *const uplo, const char *const trans, const int *const n, const int *const k, const T *const alpha, const T *const a, const int *const lda, const T *const beta, T *const c, const int *const ldc) {                                        \
            HTOOL_BLAS_F77(C##syrk)                                                                                                                                                                                                                                                      \
            (uplo, trans, n, k, alpha, a, lda, beta, c, ldc);                                                                                                                                                                                                                            \
        }                                                                                                                                                                                                                                                                                \
        template <>                                                                                                                                                                                                                                                                      \
        inline void Blas<T>::trsm(const char *side, const char *uplo, const char *transa, const char *diag, const int *m, const int *n, const T *alpha, const T *a, const int *lda, T *b, const int *ldb) {                                                                              \
            HTOOL_BLAS_F77(C##trsm)                                                                                                                                                                                                                                                      \
            (side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);                                                                                                                                                                                                                     \
        }                                                                                                                                                                                                                                                                                \
        template <>                                                                                                                                                                                                                                                                      \
        inline void Blas<T>::laswp(const int *N, T *A, const int *lda, const int *K1, const int *K2, const int *ipiv, const int *incx) {                                                                                                                                                 \
            HTOOL_BLAS_F77(C##laswp)                                                                                                                                                                                                                                                     \
            (N, A, lda, K1, K2, ipiv, incx);                                                                                                                                                                                                                                             \
        }
#    define HTOOL_GENERATE_BLAS_COMPLEX(C, T, B, U)                                                                                                                                                                                                                                      \
        HTOOL_GENERATE_BLAS(C, T)                                                                                                                                                                                                                                                        \
        HTOOL_GENERATE_GEMM(B, U)                                                                                                                                                                                                                                                        \
        HTOOL_GENERATE_GEMM_COMPLEX(C, T)                                                                                                                                                                                                                                                \
        HTOOL_GENERATE_BLAS(B, U)                                                                                                                                                                                                                                                        \
        template <>                                                                                                                                                                                                                                                                      \
        inline U Blas<U>::nrm2(const int *const n, const U *const x, const int *const incx) {                                                                                                                                                                                            \
            return HTOOL_BLAS_F77(B##nrm2)(n, x, incx);                                                                                                                                                                                                                                  \
        }                                                                                                                                                                                                                                                                                \
        template <>                                                                                                                                                                                                                                                                      \
        inline U Blas<T>::nrm2(const int *const n, const T *const x, const int *const incx) {                                                                                                                                                                                            \
            return HTOOL_BLAS_F77(B##C##nrm2)(n, x, incx);                                                                                                                                                                                                                               \
        }                                                                                                                                                                                                                                                                                \
        template <>                                                                                                                                                                                                                                                                      \
        inline void Blas<T>::hemv(const char *const uplo, const int *const n, const T *const alpha, const T *const a, const int *const lda, const T *const x, const int *const incx, const T *const beta, T *const y, const int *const incy) {                                           \
            HTOOL_BLAS_F77(C##hemv)                                                                                                                                                                                                                                                      \
            (uplo, n, alpha, a, lda, x, incx, beta, y, incy);                                                                                                                                                                                                                            \
        }                                                                                                                                                                                                                                                                                \
        template <>                                                                                                                                                                                                                                                                      \
        inline void Blas<T>::hemm(const char *const side, const char *const uplo, const int *const m, const int *const n, const T *const alpha, const T *const a, const int *const lda, const T *const b, const int *const ldb, const T *const beta, T *const c, const int *const ldc) { \
            HTOOL_BLAS_F77(C##hemm)                                                                                                                                                                                                                                                      \
            (side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);                                                                                                                                                                                                                     \
        }                                                                                                                                                                                                                                                                                \
        template <>                                                                                                                                                                                                                                                                      \
        inline void Blas<T>::herk(const char *const uplo, const char *const trans, const int *const n, const int *const k, const U *const alpha, const T *const a, const int *const lda, const U *const beta, T *const c, const int *const ldc) {                                        \
            HTOOL_BLAS_F77(C##herk)                                                                                                                                                                                                                                                      \
            (uplo, trans, n, k, alpha, a, lda, beta, c, ldc);                                                                                                                                                                                                                            \
        }

HTOOL_GENERATE_BLAS_COMPLEX(c, std::complex<float>, s, float)
HTOOL_GENERATE_BLAS_COMPLEX(z, std::complex<double>, d, double)

} // namespace htool
#endif // __cplusplus
#endif // HTOOL_BLAS_HPP
