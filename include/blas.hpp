/*
   This file is part of HPDDM.

   Author(s): Pierre Jolivet <pierre.jolivet@enseeiht.fr>
        Date: 2015-10-21

   Copyright (C) 2015      Eidgenössische Technische Hochschule Zürich
                 2016-     Centre National de la Recherche Scientifique

   HPDDM is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published
   by the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   HPDDM is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public License
   along with HPDDM.  If not, see <http://www.gnu.org/licenses/>.
 */
#include <complex>
#ifndef BLAS_HPP
#define BLAS_HPP


#if defined(__powerpc__) || defined(INTEL_MKL_VERSION)
# define HPDDM_F77(func) func
#else
# define HPDDM_F77(func) func ## _
#endif


#define HPDDM_GENERATE_EXTERN_BLAS(C, T)                                                                     \
void    HPDDM_F77(C ## gemv)(const char*, const int*, const int*, const T*,                                  \
                             const T*, const int*, const T*, const int*,                                     \
                             const T*, T*, const int*);

#define HPDDM_GENERATE_EXTERN_BLAS_COMPLEX(C, T, B, U)                                                       \
HPDDM_GENERATE_EXTERN_BLAS(B, U)                                                                             \
HPDDM_GENERATE_EXTERN_BLAS(C, T)


extern "C" {

HPDDM_GENERATE_EXTERN_BLAS_COMPLEX(z, std::complex<double>, d, double)

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

};
# define HPDDM_GENERATE_BLAS(C, T)                                                                           \
template<>                                                                                                   \
inline void Blas<T>::gemv(const char* const trans, const int* const m, const int* const n,                   \
                          const T* const alpha, const T* const a, const int* const lda, const T* const b,    \
                          const int* const ldb, const T* const beta, T* const c, const int* const ldc) {     \
    HPDDM_F77(C ## gemv)(trans, m, n, alpha, a, lda, b, ldb, beta, c, ldc);                                  \
}                                                                                                            \

# define HPDDM_GENERATE_BLAS_COMPLEX(C, T, B, U)                                                             \
HPDDM_GENERATE_BLAS(C, T)                                                                                    \

HPDDM_GENERATE_BLAS_COMPLEX(z, std::complex<double>, d, double)

} // HPDDM

#endif // _HPDDM_BLAS_
