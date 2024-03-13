#ifndef HTOOL_VIRTUAL_COARSE_SPACE_BUILDER_HPP
#define HTOOL_VIRTUAL_COARSE_SPACE_BUILDER_HPP

namespace htool {

template <typename CoefficientPrecision>
class VirtualCoarseSpaceBuilder {
  public:
    virtual Matrix<CoefficientPrecision> build_coarse_space() = 0;
    virtual ~VirtualCoarseSpaceBuilder() {}
};

} // namespace htool

#endif
