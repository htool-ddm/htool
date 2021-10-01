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

#include "../clustering/bounding_box_1.hpp"
#include "../clustering/pca.hpp"
#include "../lrmat/SVD.hpp"
#include "../lrmat/fullACA.hpp"
#include "../lrmat/lrmat.hpp"
#include "../lrmat/sympartialACA.hpp"
#include "../types/hmatrix.hpp"

#if defined(__clang__)
#    pragma clang diagnostic pop
#elif defined(__GNUC__) || defined(__GNUG__)
#    pragma GCC diagnostic pop
#endif

#endif