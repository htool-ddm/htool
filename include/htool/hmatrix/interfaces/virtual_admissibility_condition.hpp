#ifndef HTOOL_HMATRIX_VIRTUAL_BLOCKS_ADMISSIBILITY_CONDITIONS_HPP
#define HTOOL_HMATRIX_VIRTUAL_BLOCKS_ADMISSIBILITY_CONDITIONS_HPP

#include "../../basic_types/vector.hpp"      // for norm2
#include "../../clustering/cluster_node.hpp" // for Cluster

namespace htool {

template <typename CoordinatePrecision>
class VirtualAdmissibilityCondition {
  public:
    virtual bool ComputeAdmissibility(const Cluster<CoordinatePrecision> &target, const Cluster<CoordinatePrecision> &source, double eta) const = 0;
    virtual ~VirtualAdmissibilityCondition() {}
};

// Rjasanow - Steinbach (3.15) p111 Chap Approximation of Boundary Element Matrices
template <typename CoordinatePrecision>
class RjasanowSteinbach final : public VirtualAdmissibilityCondition<CoordinatePrecision> {
  public:
    bool ComputeAdmissibility(const Cluster<CoordinatePrecision> &target, const Cluster<CoordinatePrecision> &source, double eta) const override {
        bool admissible = 2 * std::min(target.get_radius(), source.get_radius()) < eta * std::max((norm2(target.get_center() - source.get_center()) - target.get_radius() - source.get_radius()), CoordinatePrecision(0));
        return admissible;
    }
};

} // namespace htool
#endif
