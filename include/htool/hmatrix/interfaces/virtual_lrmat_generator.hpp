#ifndef HTOOL_VIRTUAL_LRMAT_GENERATOR_HPP
#define HTOOL_VIRTUAL_LRMAT_GENERATOR_HPP

#include "../../clustering/cluster_node.hpp" // for Cluster
#include "../../matrix/matrix.hpp"           // for Matrix
#include "../../misc/misc.hpp"               // for underlying_type
#include "virtual_generator.hpp"             // for VirtualGenerator

namespace htool {

template <typename CoefficientPrecision, typename CoordinatesPrecision = underlying_type<CoefficientPrecision>>
class VirtualLowRankGenerator {
  public:
    VirtualLowRankGenerator() {}

    // C style
    virtual void copy_low_rank_approximation(const VirtualInternalGenerator<CoefficientPrecision> &A, const Cluster<CoordinatesPrecision> &t, const Cluster<CoordinatesPrecision> &s, underlying_type<CoefficientPrecision> epsilon, int &rank, Matrix<CoefficientPrecision> &U, Matrix<CoefficientPrecision> &V) const = 0;

    virtual bool is_htool_owning_data() const { return true; }
    virtual ~VirtualLowRankGenerator() {}
};

} // namespace htool

#endif
