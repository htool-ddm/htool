#ifndef HTOOL_VIRTUAL_OPERATOR_SPACE_BUILDER_HPP
#define HTOOL_VIRTUAL_OPERATOR_SPACE_BUILDER_HPP

#include "../../matrix/matrix.hpp"

namespace htool {

template <typename CoefficientPrecision>
class VirtualCoarseOperatorBuilder {
  public:
    virtual Matrix<CoefficientPrecision> build_coarse_operator(int nb_rows, int nb_cols, CoefficientPrecision **Z) = 0;
    virtual ~VirtualCoarseOperatorBuilder() {}
};

} // namespace htool

#endif
