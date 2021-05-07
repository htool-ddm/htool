#ifndef HTOOL_LAPACK_HPP
#define HTOOL_LAPACK_HPP

#include "../misc/define.hpp"

#if defined(__powerpc__) || defined(INTEL_MKL_VERSION)
#    define HTOOL_LAPACK_F77(func) func
#else
#    define HTOOL_LAPACK_F77(func) func##_
#endif

#define HTOOL_GENERATE_EXTERN_LAPACK_COMPLEX(C, T, B, U)                                                                                                                       \
    void HTOOL_LAPACK_F77(B##gesvd)(const char *, const char *, const int *, const int *, U *, const int *, U *, U *, const int *, U *, const int *, U *, const int *, int *); \
    void HTOOL_LAPACK_F77(C##gesvd)(const char *, const char *, const int *, const int *, T *, const int *, U *, T *, const int *, T *, const int *, T *, const int *, U *, int *);

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
    /* Function: gesvd
     *  computes the singular value decomposition (SVD). */
    static void gesvd(const char *, const char *, const int *, const int *, K *, const int *, underlying_type<K> *, K *, const int *, K *, const int *, K *, const int *, underlying_type<K> *, int *);
};

#    define HTOOL_GENERATE_LAPACK_COMPLEX(C, T, B, U)                                                                                                                                                                             \
        template <>                                                                                                                                                                                                               \
        inline void Lapack<U>::gesvd(const char *jobu, const char *jobvt, const int *m, const int *n, U *a, const int *lda, U *s, U *u, const int *ldu, U *vt, const int *ldvt, U *work, const int *lwork, U *, int *info) {      \
            HTOOL_LAPACK_F77(B##gesvd)                                                                                                                                                                                            \
            (jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, info);                                                                                                                                                  \
        }                                                                                                                                                                                                                         \
        template <>                                                                                                                                                                                                               \
        inline void Lapack<T>::gesvd(const char *jobu, const char *jobvt, const int *m, const int *n, T *a, const int *lda, U *s, T *u, const int *ldu, T *vt, const int *ldvt, T *work, const int *lwork, U *rwork, int *info) { \
            HTOOL_LAPACK_F77(C##gesvd)                                                                                                                                                                                            \
            (jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, rwork, info);                                                                                                                                           \
        }

HTOOL_GENERATE_LAPACK_COMPLEX(c, std::complex<float>, s, float)
HTOOL_GENERATE_LAPACK_COMPLEX(z, std::complex<double>, d, double)
} // namespace htool
#endif // __cplusplus
#endif // HTOOL_LAPACK_HPP
