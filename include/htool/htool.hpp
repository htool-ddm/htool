#ifndef HTOOL_HPP
#define HTOOL_HPP



#ifndef HTOOL_MKL
# ifdef INTEL_MKL_VERSION
#  define HTOOL_MKL           1
# else
#  define HTOOL_MKL           0
# endif
#endif

// #include "schwarz.hpp"
#include "blas.hpp"
#include "cluster.hpp"
#include "export.hpp"
#include "fullACA.hpp"
#include "geometry.hpp"
#include "hmatrix.hpp"
#include "lapack.hpp"
#include "lrmat.hpp"
#include "matrix.hpp"
#include "output.hpp"
#include "parametres.hpp"
#include "partialACA.hpp"
#include "point.hpp"
#include "preconditioner.hpp"
#include "user.hpp"
#include "vector.hpp"
#include "wrapper_hpddm.hpp"
#include "wrapper_mpi.hpp"

#endif
