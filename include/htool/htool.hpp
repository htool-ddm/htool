#ifndef HTOOL_HTOOL_HPP
#define HTOOL_HTOOL_HPP

#include "htool_version.hpp"
#include "misc/define.hpp"

#include "basic_types/vector.hpp"
#include "clustering/cluster_output.hpp"
#include "clustering/tree_builder/tree_builder.hpp"
#include "distributed_operator/distributed_operator.hpp"
#include "distributed_operator/linalg.hpp"
#include "distributed_operator/utility.hpp"
#include "hmatrix/hmatrix.hpp"
#include "hmatrix/hmatrix_distributed_output.hpp"
#include "hmatrix/interfaces/virtual_generator.hpp"
#include "hmatrix/linalg.hpp"
#include "hmatrix/lrmat/SVD.hpp"
#include "hmatrix/lrmat/fullACA.hpp"
#include "hmatrix/lrmat/linalg.hpp"
#include "hmatrix/lrmat/lrmat.hpp"
#include "hmatrix/lrmat/partialACA.hpp"
#include "hmatrix/lrmat/recompressed_low_rank_generator.hpp"
#include "hmatrix/lrmat/sympartialACA.hpp"
#include "matrix/linalg.hpp"
#include "matrix/matrix.hpp"
#include "matrix/matrix_view.hpp"
#include "misc/misc.hpp"
#include "misc/user.hpp"

#ifdef HTOOL_WITH_HPDDM
#    include "solvers/ddm.hpp"
#    include "solvers/geneo/coarse_operator_builder.hpp"
#    include "solvers/geneo/coarse_space_builder.hpp"
#    include "solvers/utility.hpp"
#    include "wrappers/wrapper_hpddm.hpp"
#endif
#include "wrappers/wrapper_blas.hpp"
#include "wrappers/wrapper_mpi.hpp"

#endif
