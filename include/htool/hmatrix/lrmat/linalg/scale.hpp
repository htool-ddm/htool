#ifndef HTOOL_LRMAT_LINALG_SCALE_HPP
#define HTOOL_LRMAT_LINALG_SCALE_HPP

#include "../../../matrix/linalg/scale.hpp"
#include "../lrmat.hpp"
namespace htool {

template <typename CoefficientPrecision>
void scale(CoefficientPrecision da, LowRankMatrix<CoefficientPrecision> &lrmat) {
    if (lrmat.get_U().nb_rows() > lrmat.get_U().nb_cols()) {
        scale(da, lrmat.get_U());
    } else {
        scale(da, lrmat.get_V());
    }
}

} // namespace htool

#endif
