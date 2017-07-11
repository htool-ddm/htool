#ifndef HTOOL_HPP
#define HTOOL_HPP

#include "config.h"
#include "blas.hpp"
#include "cluster.hpp"
#include "export.hpp"
#include "hmatrix.hpp"
#include "loading.hpp"
#include "lrmat.hpp"
#include "matrix.hpp"
#include "parametres.hpp"
#include "point.hpp"
#include "user.hpp"

#if HTOOL_WITH_GUI==ON  && defined GLM_FOUND
#if GLM_FOUND==TRUE
#include "view.hpp"
#endif
#endif

#endif
