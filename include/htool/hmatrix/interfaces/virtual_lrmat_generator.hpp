#ifndef HTOOL_VIRTUAL_LRMAT_GENERATOR_HPP
#define HTOOL_VIRTUAL_LRMAT_GENERATOR_HPP

#include "../../matrix/matrix.hpp"
#include "../../misc/misc.hpp"
#include "virtual_generator.hpp"
#include <cassert>
#include <iterator>

namespace htool {

template <typename CoefficientPrecision, typename CoordinatesPrecision>
class VirtualLowRankGenerator {
  public:
    VirtualLowRankGenerator() {}

    // C style
    virtual void copy_low_rank_approximation(const VirtualGenerator<CoefficientPrecision> &A, const Cluster<CoordinatesPrecision> &t, const Cluster<CoordinatesPrecision> &s, underlying_type<CoefficientPrecision> epsilon, int &rank, Matrix<CoefficientPrecision> &U, Matrix<CoefficientPrecision> &V) const = 0;

    virtual bool is_htool_owning_data() const { return true; }
    virtual ~VirtualLowRankGenerator() {}
};

} // namespace htool

#endif
