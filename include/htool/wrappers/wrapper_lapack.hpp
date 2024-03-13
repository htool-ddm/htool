#ifndef HTOOL_LAPACK_HPP
#define HTOOL_LAPACK_HPP

#include "../misc/define.hpp"

#if defined(__powerpc__) || defined(INTEL_MKL_VERSION)
#    define HTOOL_LAPACK_F77(func) func
#else
#    define HTOOL_LAPACK_F77(func) func##_
#endif

#define HTOOL_GENERATE_EXTERN_LAPACK(C, T)                                                                     \
    void HTOOL_LAPACK_F77(C##geqrf)(const int *, const int *, T *, const int *, T *, T *, const int *, int *); \
    void HTOOL_LAPACK_F77(C##gelqf)(const int *, const int *, T *, const int *, T *, T *, const int *, int *);

// const int *itype, const char *jobz, const char *UPLO, const int *n, T *A, const int *lda, T *B, const int *ldb, T *W, T *work, const int *lwork, U *, int *info)
#define HTOOL_GENERATE_EXTERN_LAPACK_COMPLEX(C, T, B, U)                                                                                                                                     \
    HTOOL_GENERATE_EXTERN_LAPACK(B, U)                                                                                                                                                       \
    HTOOL_GENERATE_EXTERN_LAPACK(C, T)                                                                                                                                                       \
    void HTOOL_LAPACK_F77(B##gesvd)(const char *, const char *, const int *, const int *, U *, const int *, U *, U *, const int *, U *, const int *, U *, const int *, int *);               \
    void HTOOL_LAPACK_F77(C##gesvd)(const char *, const char *, const int *, const int *, T *, const int *, U *, T *, const int *, T *, const int *, T *, const int *, U *, int *);          \
    void HTOOL_LAPACK_F77(B##ggev)(const char *, const char *, const int *, U *, const int *, U *, const int *, U *, U *, U *, U *, const int *, U *, const int *, U *, const int *, int *); \
    void HTOOL_LAPACK_F77(C##ggev)(const char *, const char *, const int *, T *, const int *, T *, const int *, T *, T *, T *, const int *, T *, const int *, T *, const int *, U *, int *); \
    void HTOOL_LAPACK_F77(B##sygv)(const int *, const char *, const char *, const int *, U *, const int *, U *, const int *, U *, U *, const int *, int *);                                  \
    void HTOOL_LAPACK_F77(C##hegv)(const int *, const char *, const char *, const int *, T *, const int *, T *, const int *, U *, T *, const int *, U *, int *);                             \
    void HTOOL_LAPACK_F77(B##ormlq)(const char *, const char *, const int *, const int *, const int *, const U *, const int *, const U *, U *, const int *, U *, const int *, int *);        \
    void HTOOL_LAPACK_F77(C##unmlq)(const char *, const char *, const int *, const int *, const int *, const T *, const int *, const T *, T *, const int *, T *, const int *, int *);        \
    void HTOOL_LAPACK_F77(B##ormqr)(const char *, const char *, const int *, const int *, const int *, const U *, const int *, const U *, U *, const int *, U *, const int *, int *);        \
    void HTOOL_LAPACK_F77(C##unmqr)(const char *, const char *, const int *, const int *, const int *, const T *, const int *, const T *, T *, const int *, T *, const int *, int *);

#if !defined(PETSC_HAVE_BLASLAPACK)
#    ifndef _MKL_H_
#        ifdef __cplusplus
extern "C" {
HTOOL_GENERATE_EXTERN_LAPACK_COMPLEX(c, std::complex<float>, s, float)
HTOOL_GENERATE_EXTERN_LAPACK_COMPLEX(z, std::complex<double>, d, double)
}
#        else
HTOOL_GENERATE_EXTERN_LAPACK_COMPLEX(c, void, s, float)
HTOOL_GENERATE_EXTERN_LAPACK_COMPLEX(z, void, d, double)
#        endif // __cplusplus
#    endif     // _MKL_H_
#endif

#ifdef __cplusplus
namespace htool {
/* Class: Lapack
 *
 *  A class that wraps some LAPACK routines for dense linear algebra.
 *
 * Template Parameter:
 *    K              - Scalar type. */
template <class K>
struct Lapack {
    /* Function: geqrf
     *  Computes a QR decomposition of a rectangular matrix. */
    static void geqrf(const int *, const int *, K *, const int *, K *, K *, const int *, int *);
    /* Function: gelqf
     *  Computes a LQ decomposition of a rectangular matrix. */
    static void gelqf(const int *, const int *, K *, const int *, K *, K *, const int *, int *);
    /* Function: mqr
     *  Multiplies a matrix by an orthogonal or unitary matrix obtained with geqrf. */
    static void mqr(const char *, const char *, const int *, const int *, const int *, const K *, const int *, const K *, K *, const int *, K *, const int *, int *);
    /* Function: mlq
     *  Multiplies a matrix by an orthogonal or unitary matrix obtained with gelqf. */
    static void mlq(const char *, const char *, const int *, const int *, const int *, const K *, const int *, const K *, K *, const int *, K *, const int *, int *);
    /* Function: gesvd
     *  computes the singular value decomposition (SVD). */
    static void gesvd(const char *, const char *, const int *, const int *, K *, const int *, underlying_type<K> *, K *, const int *, K *, const int *, K *, const int *, underlying_type<K> *, int *);
    /* Function: ggev
     *  Computes the eigenvalues and (optionally) the eigenvectors of a nonsymmetric generalized eigenvalue problem. */
    static void ggev(const char *, const char *, const int *, K *, const int *, K *, const int *, K *, K *, K *, K *, const int *, K *, const int *, K *, const int *, underlying_type<K> *, int *);
    /* Function: gv
     *  Computes the eigenvalues and (optionally) the eigenvectors of a hermitian/symetric generalized eigenvalue problem. */
    static void gv(const int *, const char *, const char *, const int *, K *, const int *, K *, const int *, underlying_type<K> *, K *, const int *, underlying_type<K> *, int *);
};

#    define HTOOL_GENERATE_LAPACK(C, T)                                                                                                \
        template <>                                                                                                                    \
        inline void Lapack<T>::geqrf(const int *m, const int *n, T *a, const int *lda, T *tau, T *work, const int *lwork, int *info) { \
            HTOOL_LAPACK_F77(C##geqrf)                                                                                                 \
            (m, n, a, lda, tau, work, lwork, info);                                                                                    \
        }                                                                                                                              \
        template <>                                                                                                                    \
        inline void Lapack<T>::gelqf(const int *m, const int *n, T *a, const int *lda, T *tau, T *work, const int *lwork, int *info) { \
            HTOOL_LAPACK_F77(C##gelqf)                                                                                                 \
            (m, n, a, lda, tau, work, lwork, info);                                                                                    \
        }

#    define HTOOL_GENERATE_LAPACK_COMPLEX(C, T, B, U)                                                                                                                                                                                                           \
        HTOOL_GENERATE_LAPACK(B, U)                                                                                                                                                                                                                             \
        HTOOL_GENERATE_LAPACK(C, T)                                                                                                                                                                                                                             \
        template <>                                                                                                                                                                                                                                             \
        inline void Lapack<U>::mqr(const char *side, const char *trans, const int *m, const int *n, const int *k, const U *a, const int *lda, const U *tau, U *c, const int *ldc, U *work, const int *lwork, int *info) {                                       \
            HTOOL_LAPACK_F77(B##ormqr)                                                                                                                                                                                                                          \
            (side, trans, m, n, k, a, lda, tau, c, ldc, work, lwork, info);                                                                                                                                                                                     \
        }                                                                                                                                                                                                                                                       \
        template <>                                                                                                                                                                                                                                             \
        inline void Lapack<T>::mqr(const char *side, const char *trans, const int *m, const int *n, const int *k, const T *a, const int *lda, const T *tau, T *c, const int *ldc, T *work, const int *lwork, int *info) {                                       \
            HTOOL_LAPACK_F77(C##unmqr)                                                                                                                                                                                                                          \
            (side, trans, m, n, k, a, lda, tau, c, ldc, work, lwork, info);                                                                                                                                                                                     \
        }                                                                                                                                                                                                                                                       \
        template <>                                                                                                                                                                                                                                             \
        inline void Lapack<U>::mlq(const char *side, const char *trans, const int *m, const int *n, const int *k, const U *a, const int *lda, const U *tau, U *c, const int *ldc, U *work, const int *lwork, int *info) {                                       \
            HTOOL_LAPACK_F77(B##ormlq)                                                                                                                                                                                                                          \
            (side, trans, m, n, k, a, lda, tau, c, ldc, work, lwork, info);                                                                                                                                                                                     \
        }                                                                                                                                                                                                                                                       \
        template <>                                                                                                                                                                                                                                             \
        inline void Lapack<T>::mlq(const char *side, const char *trans, const int *m, const int *n, const int *k, const T *a, const int *lda, const T *tau, T *c, const int *ldc, T *work, const int *lwork, int *info) {                                       \
            HTOOL_LAPACK_F77(C##unmlq)                                                                                                                                                                                                                          \
            (side, trans, m, n, k, a, lda, tau, c, ldc, work, lwork, info);                                                                                                                                                                                     \
        }                                                                                                                                                                                                                                                       \
        template <>                                                                                                                                                                                                                                             \
        inline void Lapack<U>::gesvd(const char *jobu, const char *jobvt, const int *m, const int *n, U *a, const int *lda, U *s, U *u, const int *ldu, U *vt, const int *ldvt, U *work, const int *lwork, U *, int *info) {                                    \
            HTOOL_LAPACK_F77(B##gesvd)                                                                                                                                                                                                                          \
            (jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, info);                                                                                                                                                                                \
        }                                                                                                                                                                                                                                                       \
        template <>                                                                                                                                                                                                                                             \
        inline void Lapack<T>::gesvd(const char *jobu, const char *jobvt, const int *m, const int *n, T *a, const int *lda, U *s, T *u, const int *ldu, T *vt, const int *ldvt, T *work, const int *lwork, U *rwork, int *info) {                               \
            HTOOL_LAPACK_F77(C##gesvd)                                                                                                                                                                                                                          \
            (jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, rwork, info);                                                                                                                                                                         \
        }                                                                                                                                                                                                                                                       \
        template <>                                                                                                                                                                                                                                             \
        inline void Lapack<U>::ggev(const char *jobvl, const char *jobvr, const int *n, U *a, const int *lda, U *b, const int *ldb, U *alphar, U *alphai, U *beta, U *vl, const int *ldvl, U *vr, const int *ldvr, U *work, const int *lwork, U *, int *info) { \
            HTOOL_LAPACK_F77(B##ggev)                                                                                                                                                                                                                           \
            (jobvl, jobvr, n, a, lda, b, ldb, alphar, alphai, beta, vl, ldvl, vr, ldvr, work, lwork, info);                                                                                                                                                     \
        }                                                                                                                                                                                                                                                       \
        template <>                                                                                                                                                                                                                                             \
        inline void Lapack<T>::ggev(const char *jobvl, const char *jobvr, const int *n, T *a, const int *lda, T *b, const int *ldb, T *alpha, T *, T *beta, T *vl, const int *ldvl, T *vr, const int *ldvr, T *work, const int *lwork, U *rwork, int *info) {   \
            HTOOL_LAPACK_F77(C##ggev)                                                                                                                                                                                                                           \
            (jobvl, jobvr, n, a, lda, b, ldb, alpha, beta, vl, ldvl, vr, ldvr, work, lwork, rwork, info);                                                                                                                                                       \
        }                                                                                                                                                                                                                                                       \
        template <>                                                                                                                                                                                                                                             \
        inline void Lapack<U>::gv(const int *itype, const char *jobz, const char *uplo, const int *n, U *a, const int *lda, U *b, const int *ldb, U *w, U *work, const int *lwork, U *, int *info) {                                                            \
            HTOOL_LAPACK_F77(B##sygv)                                                                                                                                                                                                                           \
            (itype, jobz, uplo, n, a, lda, b, ldb, w, work, lwork, info);                                                                                                                                                                                       \
        }                                                                                                                                                                                                                                                       \
        template <>                                                                                                                                                                                                                                             \
        inline void Lapack<T>::gv(const int *itype, const char *jobz, const char *uplo, const int *n, T *a, const int *lda, T *b, const int *ldb, U *w, T *work, const int *lwork, U *rwork, int *info) {                                                       \
            HTOOL_LAPACK_F77(C##hegv)                                                                                                                                                                                                                           \
            (itype, jobz, uplo, n, a, lda, b, ldb, w, work, lwork, rwork, info);                                                                                                                                                                                \
        }

HTOOL_GENERATE_LAPACK_COMPLEX(c, std::complex<float>, s, float)
HTOOL_GENERATE_LAPACK_COMPLEX(z, std::complex<double>, d, double)
} // namespace htool
#endif // __cplusplus
#endif // HTOOL_LAPACK_HPP
