#ifndef HTOOL_LAPACK_HPP
#define HTOOL_LAPACK_HPP


#if defined(__powerpc__) || defined(INTEL_MKL_VERSION)
# define HTOOL_F77(func) func
#else
# define HTOOL_F77(func) func ## _
#endif


#define HTOOL_GENERATE_EXTERN_LAPACK(C, T, U, SYM, ORT)                        \
void HTOOL_F77(C ## potrf)(const char*, const int*, T*, const int*, int*);     \
void HTOOL_F77(C ## potrs)(const char*, const int*, const int*, const T*, const int*, T*, const int*, int*);                                                         \


#define HTOOL_GENERATE_EXTERN_LAPACK_COMPLEX(C, T, B, U)                       \
HTOOL_GENERATE_EXTERN_LAPACK(B, U, U, sy, or)                                  \
HTOOL_GENERATE_EXTERN_LAPACK(C, T, U, he, un)                                  \

extern "C" {
HTOOL_GENERATE_EXTERN_LAPACK_COMPLEX(c, std::complex<float>, s, float)
HTOOL_GENERATE_EXTERN_LAPACK_COMPLEX(z, std::complex<double>, d, double)
}

namespace htool{
template<class K>
struct Lapack {

  static void potrf(const char*, const int*, K*, const int*, int*);
  static void potrs(const char*, const int*, const int*, const K*, const int*, K*, const int*, int*);
};


# define HTOOL_GENERATE_LAPACK(C, T, B, U, SYM, ORT)                           \
template<>                                                                     \
inline void Lapack<T>::potrf(const char* uplo, const int* n, T* a, const int* lda, int* info) {                                                                        \
    HTOOL_F77(C ## potrf)(uplo, n, a, lda, info);                              \
}                                                                              \
template<>                                                                     \
inline void Lapack<T>::potrs(const char* uplo, const int* n, const int* nrhs, const T* a, const int* lda, T* b, const int* ldb, int* info) {                          \
    HTOOL_F77(C ## potrs)(uplo, n, nrhs, a, lda, b, ldb, info);                \
}

# define HTOOL_GENERATE_LAPACK_COMPLEX(C, T, B, U)                             \
HTOOL_GENERATE_LAPACK(B, U, B, U, sy, or)                                      \
HTOOL_GENERATE_LAPACK(C, T, B, U, he, un)                                      \

HTOOL_GENERATE_LAPACK_COMPLEX(c, std::complex<float>, s, float)
HTOOL_GENERATE_LAPACK_COMPLEX(z, std::complex<double>, d, double)
}


#endif
