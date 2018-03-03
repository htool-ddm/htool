#ifndef HTOOL_HTOOL_HPP
#define HTOOL_HTOOL_HPP



#ifndef HTOOL_MKL
# ifdef INTEL_MKL_VERSION
#  define HTOOL_MKL           1
# else
#  define HTOOL_MKL           0
# endif
#endif


#include "clustering/cluster.hpp"
#include "clustering/cluster_tree.hpp"

#include "input_output/export.hpp"
#include "input_output/geometry.hpp"
#include "input_output/output.hpp"

#include "lrmat/lrmat.hpp"
#include "lrmat/fullACA.hpp"
#include "lrmat/partialACA.hpp"

#include "misc/infos.hpp"
#include "misc/parametres.hpp"
#include "misc/user.hpp"

#include "solvers/solver.hpp"
#include "solvers/schwarz.hpp"

#include "types/hmatrix.hpp"
#include "types/matrix.hpp"
#include "types/point.hpp"
#include "types/vector.hpp"

#include "wrappers/wrapper_blas.hpp"
#include "wrappers/wrapper_hpddm.hpp"
#include "wrappers/wrapper_mpi.hpp"
#include "wrappers/wrapper_lapack.hpp"


#endif
