#ifndef HTOOL_QUADRATURES_DUNAVANT_HPP
#define HTOOL_QUADRATURES_DUNAVANT_HPP

#include "quadrature.hpp"
#include <cstddef>
#include <string>
namespace htool {

template <typename T>
struct DunavantRule {
    inline static const std::string name = "Dunavant";
    std::size_t degree;
    std::size_t nb_points;
    const QuadPoint<T, 2> *quad_points;
};

// ---------- Degree 1 (1 point) ----------
template <typename T>
constexpr QuadPoint<T, 2> dunavant_deg1[] = {
    {{T(1.0) / T(3.0), T(1.0) / T(3.0)}, T(0.5)}};

// ---------- Degree 2 (3 points) ----------
template <typename T>
constexpr QuadPoint<T, 2> dunavant_deg2[] = {
    {{T(1.0) / T(6.0), T(1.0) / T(6.0)}, T(1.0) / T(6.0)},
    {{T(2.0) / T(3.0), T(1.0) / T(6.0)}, T(1.0) / T(6.0)},
    {{T(1.0) / T(6.0), T(2.0) / T(3.0)}, T(1.0) / T(6.0)},
};

// ---------- Degree 3 (4 points) ----------
template <typename T>
constexpr QuadPoint<T, 2> dunavant_deg3[] = {
    {{T(1.0) / T(3.0), T(1.0) / T(3.0)}, T(-9.0) / T(32.0)},
    {{T(0.2), T(0.2)}, T(25.0) / T(96.0)},
    {{T(0.6), T(0.2)}, T(25.0) / T(96.0)},
    {{T(0.2), T(0.6)}, T(25.0) / T(96.0)},
};

// // ---------- Degree 4 (6 points) ----------
// template <typename T>
// constexpr QuadPoint<T, 2> dunavant_deg4[] = {
//     {{T(0.445948490915965), T(0.108103018168070)}, T(0.223381589678011)},
//     {{T(0.108103018168070), T(0.445948490915965)}, T(0.223381589678011)},
//     {{T(0.445948490915965), T(0.445948490915965)}, T(0.223381589678011)},
//     {{T(0.091576213509771), T(0.816847572980459)}, T(0.109951743655322)},
//     {{T(0.816847572980459), T(0.091576213509771)}, T(0.109951743655322)},
//     {{T(0.091576213509771), T(0.091576213509771)}, T(0.109951743655322)},
// };

// // ---------- Degree 5 (7 points) ----------
// template <typename T>
// constexpr QuadPoint<T, 2> dunavant_deg5[] = {
//     {{T(1.0) / T(3.0), T(1.0) / T(3.0)}, T(0.225000000000000)},
//     {{T(0.470142064105115), T(0.470142064105115)}, T(0.132394152788506)},
//     {{T(0.470142064105115), T(0.059715871789770)}, T(0.132394152788506)},
//     {{T(0.059715871789770), T(0.470142064105115)}, T(0.132394152788506)},
//     {{T(0.101286507323456), T(0.101286507323456)}, T(0.125939180544827)},
//     {{T(0.101286507323456), T(0.797426985353087)}, T(0.125939180544827)},
//     {{T(0.797426985353087), T(0.101286507323456)}, T(0.125939180544827)},
// };

// // ---------- Degree 6 (12 points) ----------
// constexpr double dunavant_deg6_x[] = {
//     0.249286745170910,
//     0.249286745170910,
//     0.501426509658179,
//     0.063089014491502,
//     0.063089014491502,
//     0.873821971016996,
//     0.310352451033785,
//     0.636502499121399,
//     0.053145049844816,
//     0.636502499121399,
//     0.053145049844816,
//     0.310352451033785};

// constexpr double dunavant_deg6_y[] = {
//     0.249286745170910,
//     0.501426509658179,
//     0.249286745170910,
//     0.063089014491502,
//     0.873821971016996,
//     0.063089014491502,
//     0.636502499121399,
//     0.053145049844816,
//     0.310352451033785,
//     0.310352451033785,
//     0.636502499121399,
//     0.053145049844816};

// constexpr double dunavant_deg6_w[] = {
//     0.058393137863189,
//     0.058393137863189,
//     0.058393137863189,
//     0.025422453185103,
//     0.025422453185103,
//     0.025422453185103,
//     0.041425537809187,
//     0.041425537809187,
//     0.041425537809187,
//     0.041425537809187,
//     0.041425537809187,
//     0.041425537809187};

template <typename T>
constexpr std::array<DunavantRule<T>, 3> dunavant_rules = {{
    {1, 1, dunavant_deg1<T>},
    {2, 3, dunavant_deg2<T>},
    {3, 4, dunavant_deg3<T>},
    // {4, 6, dunavant_deg4<T>},
    // {5, 7, dunavant_deg5<T>},
    // {6, 12, dunavant_deg6_x, dunavant_deg6_y, dunavant_deg6_w},
    // {7, 13, dunavant_deg7_x, dunavant_deg7_y, dunavant_deg7_w},
    // {8, 16, dunavant_deg8_x, dunavant_deg8_y, dunavant_deg8_w},
    // {9, 19, dunavant_deg9},
    // {10, 25, dunavant_deg10},
    // {11, 27, dunavant_deg11},
    // {12, 33, dunavant_deg12},
    // {13, 37, dunavant_deg13},
    // {14, 42, dunavant_deg14},
    // {15, 48, dunavant_deg15},
    // {16, 52, dunavant_deg16},
    // {17, 61, dunavant_deg17},
    // {18, 70, dunavant_deg18},
    // {19, 73, dunavant_deg19},
    // {20, 79, dunavant_deg20}
}};

} // namespace htool
#endif
