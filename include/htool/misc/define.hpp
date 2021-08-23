#ifndef HTOOL_DEFINE_HPP
#define HTOOL_DEFINE_HPP

/* Constants: C-style preprocessor variables
 *
 *    HTOOL_VERSION       - Version of the framework.
 *    HTOOL_MKL           - If not set to zero, Intel MKL is chosen as the linear algebra backend. */

#define HTOOL_VERSION "0.6.0"
#if defined(PETSC_HAVE_MKL) && !defined(HTOOL_MKL)
#    define HTOOL_MKL 1
#endif
#ifndef HTOOL_MKL
#    ifdef INTEL_MKL_VERSION
#        define HTOOL_MKL 1
#    else
#        define HTOOL_MKL 0
#    endif
#endif

#include <complex>

namespace htool {
template <class T>
struct underlying_type_spec {
    typedef T type;
};
template <class T>
struct underlying_type_spec<std::complex<T>> {
    typedef T type;
};
template <class T>
using underlying_type = typename underlying_type_spec<T>::type;
} // namespace htool
#endif