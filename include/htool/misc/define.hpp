#ifndef HTOOL_DEFINE_HPP
#define HTOOL_DEFINE_HPP

/* Constants: C-style preprocessor variables
 *
 *    HTOOL_VERSION       - Version of the framework.
 *    HTOOL_MKL           - If not set to zero, Intel MKL is chosen as the linear algebra backend. */

#define HTOOL_VERSION "0.5.0"
#ifndef HTOOL_MKL
#    ifdef INTEL_MKL_VERSION
#        define HTOOL_MKL 1
#    else
#        define HTOOL_MKL 0
#    endif
#endif

#endif