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
#include "clustering/ncluster.hpp"

#include "input_output/geometry.hpp"
#include "input_output/output.hpp"

#include "lrmat/lrmat.hpp"
#include "lrmat/SVD.hpp"
#include "lrmat/fullACA.hpp"
#include "lrmat/partialACA.hpp"
#include "lrmat/sympartialACA.hpp"

#include "multilrmat/multilrmat.hpp"
#include "multilrmat/multipartialACA.hpp"

#include "misc/infos.hpp"
#include "misc/parametres.hpp"
#include "misc/user.hpp"

#include "types/hmatrix.hpp"
#include "types/matrix.hpp"
#include "types/multihmatrix.hpp"
#include "types/multimatrix.hpp"
#include "types/point.hpp"
#include "types/vector.hpp"

#ifdef WITH_HPDDM
    #include "wrappers/wrapper_hpddm.hpp"
#endif
#include "wrappers/wrapper_blas.hpp"
#include "wrappers/wrapper_mpi.hpp"

#ifdef WITH_HPDDM
    #include "solvers/ddm.hpp"
    #include "solvers/proto_ddm.hpp"
#endif

#endif
