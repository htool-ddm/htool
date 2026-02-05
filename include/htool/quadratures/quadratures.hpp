#ifndef HTOOL_QUADRATURES_HPP
#define HTOOL_QUADRATURES_HPP
namespace htool {
template <typename T, int dim>
struct QuadPoint {};

template <typename T>
struct QuadPoint<T, 1> {
    T x;
    T w;
};

template <typename T>
struct QuadPoint<T, 2> {
    T x;
    T y;
    T w;
};

template <typename T>
constexpr T map_to_interval(T xi, T a, T b) {
    return (b - a) * (xi * T(0.5)) + (a + b) * T(0.5);
}

template <typename T>
constexpr T map_to_reference_interval(T x, T a, T b) {
    return (T(2) * x - (a + b)) / (b - a);
}

} // namespace htool
#endif
