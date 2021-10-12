#ifndef HTOOL_BLOCKS_ADMISSIBILITY_CONDITIONS_HPP
#define HTOOL_BLOCKS_ADMISSIBILITY_CONDITIONS_HPP

#include "../clustering/virtual_cluster.hpp"

namespace htool {

class VirtualAdmissibilityCondition {
  public:
    virtual bool ComputeAdmissibility(const VirtualCluster &target, const VirtualCluster &source, double eta) const = 0;
};

// Rjasanow - Steinbach (3.15) p111 Chap Approximation of Boundary Element Matrices
class RjasanowSteinbach final : public VirtualAdmissibilityCondition {
  public:
    bool ComputeAdmissibility(const VirtualCluster &target, const VirtualCluster &source, double eta) const override {
        bool admissible = 2 * std::min(target.get_rad(), source.get_rad()) < eta * std::max((norm2(target.get_ctr() - source.get_ctr()) - target.get_rad() - source.get_rad()), 0.);
        return admissible;
    }
};

} // namespace htool
#endif
