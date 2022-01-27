#ifndef HTOOL_DEFINE_HPP
#define HTOOL_DEFINE_HPP

/* Constants: C-style preprocessor variables
 *
 *    HTOOL_VERSION       - Version of the framework.
 *    HTOOL_MKL           - If not set to zero, Intel MKL is chosen as the linear algebra backend. */

#define HTOOL_VERSION "0.8.0"
#if defined(PETSC_HAVE_MKL_LIBS) && !defined(HTOOL_MKL)
#    define HTOOL_MKL 1
#endif
#ifndef HTOOL_MKL
#    ifdef INTEL_MKL_VERSION
#        define HTOOL_MKL 1
#    else
#        define HTOOL_MKL 0
#    endif
#endif
#endif
