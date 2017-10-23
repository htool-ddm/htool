#include <complex>
#ifndef BLAS_HPP
#define BLAS_HPP


#if defined(__powerpc__) || defined(INTEL_MKL_VERSION)
# define HTOOL_F77(func) func
#else
# define HTOOL_F77(func) func ## _
#endif


#define HTOOL_GENERATE_EXTERN_BLAS(C,T)                        \
void    HTOOL_F77(C ## gemv)(const char*, const int*, const int*, const T*,    \
const T*, const int*, const T*, const int*,       \
const T*, T*, const int*);                        \
void    HTOOL_F77(C ## gemm)(const char*, const char*, const int*, const int*, const int*,                   \
const T*, const T*, const int*, const T*, const int*,                           \
const T*, T*, const int*);                           \

#define HTOOL_GENERATE_EXTERN_BLAS_COMPLEX(C, T, B, U)\
HTOOL_GENERATE_EXTERN_BLAS(B, U)                      \
HTOOL_GENERATE_EXTERN_BLAS(C, T)                      \

#if HTOOL_MKL
# define HTOOL_GENERATE_EXTERN_GEMM3M(C, T)                                                    \
void HTOOL_F77(C ## gemm3m)(const char*, const char*, const int*, const int*, const int*,                    \
const T*, const T*, const int*, const T*, const int*,                                                 \
const T*, T*, const int*);
# define HTOOL_GENERATE_EXTERN_MKL_EXTENSIONS(C, T, B, U)                                                    \
HTOOL_GENERATE_EXTERN_GEMM3M(C, T)
#endif


extern "C" {
HTOOL_GENERATE_EXTERN_BLAS_COMPLEX(c, std::complex<float>, s, float)
HTOOL_GENERATE_EXTERN_BLAS_COMPLEX(z, std::complex<double>, d, double)
#if HTOOL_MKL
HPDDM_GENERATE_EXTERN_MKL_EXTENSIONS(c, std::complex<float>, s, float)
HPDDM_GENERATE_EXTERN_MKL_EXTENSIONS(z, std::complex<double>, d, double)
#endif
}

namespace htool {
/* Class: Blas
 *
 *  A class that wraps most of BLAS routines for dense linear algebra.
 *
 * Template Parameter:
 *    K              - Scalar type. */
template<class K>
struct Blas {

    /* Function: gemv
     *  Computes a scalar-matrix-vector product. */
    static void gemv(const char* const, const int* const, const int* const, const K* const, const K* const,
    const int* const, const K* const, const int* const, const K* const, K* const, const int* const);
    /* Function: gemm
    *  Computes a scalar-matrix-matrix product. */
    static void gemm(const char* const, const char* const, const int* const, const int* const, const int* const, const K* const, const K* const,
    const int* const, const K* const, const int* const, const K* const, K* const, const int* const);

};

# define HTOOL_GENERATE_GEMM(C, T)                   \
template<>                                           \
inline void Blas<T>::gemm(const char* const transa, const char* const transb, const int* const m,        \
const int* const n, const int* const k, const T* const alpha, const T* const a,    \
const int* const lda, const T* const b, const int* const ldb, const T* const beta, \
T* const c, const int* const ldc) {                                                \
    HTOOL_F77(C ## gemm)(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);                      \
}              \

# if !HTOOL_MKL
#  define HTOOL_GENERATE_GEMM_COMPLEX(C, T)  HTOOL_GENERATE_GEMM(C, T)
# else
#  define HTOOL_GENERATE_GEMM_COMPLEX(C, T)      \
template<>                                       \
inline void Blas<T>::gemm(const char* const transa, const char* const transb, const int* const m,        \
const int* const n, const int* const k, const T* const alpha, const T* const a,    \
const int* const lda, const T* const b, const int* const ldb, const T* const beta, \
T* const c, const int* const ldc) {                                                \
    HTOOL_F77(C ## gemm3m)(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);                    \
}
# endif


# define HTOOL_GENERATE_BLAS(C, T)                 \
template<>                                           \
inline void Blas<T>::gemv(const char* const trans, const int* const m,         \
  const int* const n, const T* const alpha, const T* const a,                  \
  const int* const lda, const T* const b, const int* const ldb,                \
  const T* const beta, T* const c, const int* const ldc) {                     \
    HTOOL_F77(C ## gemv)(trans, m, n, alpha, a, lda, b, ldb, beta, c, ldc);    \
}                                                     \

# define HTOOL_GENERATE_BLAS_COMPLEX(C, T, B,U)      \
HTOOL_GENERATE_BLAS(C,T)                             \
HTOOL_GENERATE_GEMM(B, U)                            \
HTOOL_GENERATE_GEMM_COMPLEX(C, T)                    \
HTOOL_GENERATE_BLAS(B,U)                             \

HTOOL_GENERATE_BLAS_COMPLEX(c, std::complex<float>, s, float)
HTOOL_GENERATE_BLAS_COMPLEX(z, std::complex<double>, d, double)

} // HTOOL

#endif // _HTOOL_BLAS_
