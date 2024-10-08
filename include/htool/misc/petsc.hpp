#ifndef HTOOL_PETSC_HPP
#define HTOOL_PETSC_HPP

#if defined(__clang__)
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wsign-compare"
#elif defined(__GNUC__) || defined(__GNUG__)
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wsign-compare"
#endif

#include "define.hpp"
#include "misc.hpp"

#include "../clustering/clustering.hpp"
#include "../distributed_operator/distributed_operator.hpp"
#include "../distributed_operator/implementations/partition_from_cluster.hpp"
#include "../hmatrix/hmatrix.hpp"
#include "../hmatrix/lrmat/SVD.hpp"
#include "../hmatrix/lrmat/fullACA.hpp"
#include "../hmatrix/lrmat/lrmat.hpp"
#include "../hmatrix/lrmat/sympartialACA.hpp"

#if defined(__clang__)
#    pragma clang diagnostic pop
#elif defined(__GNUC__) || defined(__GNUG__)
#    pragma GCC diagnostic pop
#endif

#endif
