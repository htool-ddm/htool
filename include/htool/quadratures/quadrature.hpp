#ifndef HTOOL_QUADRATURES_HPP
#define HTOOL_QUADRATURES_HPP
#include <array>
namespace htool {

template <typename T, int dim>
struct QuadPoint {
    std::array<T, dim> point;
    T w;
};

} // namespace htool
#endif
