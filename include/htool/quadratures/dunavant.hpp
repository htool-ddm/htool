#ifndef HTOOL_QUADRATURES_DUNAVANT_HPP
#define HTOOL_QUADRATURES_DUNAVANT_HPP

#include <array>
#include <cstddef>
namespace htool {

template <typename T>
struct DunavantRule {
    std::size_t degree;
    std::size_t nb_points;
    const T *x;
    const T *y;
    const T *w;
};

// ---------- Degree 1 (1 point) ----------
constexpr double dunavant_deg1_x[] = {1.0 / 3.0};
constexpr double dunavant_deg1_y[] = {1.0 / 3.0};
constexpr double dunavant_deg1_w[] = {0.5};

// ---------- Degree 2 (3 points) ----------
// constexpr double dunavant_deg2_x[] = {1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0};
// constexpr double dunavant_deg2_y[] = {1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0};
// constexpr double dunavant_deg2_w[] = {1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0};

constexpr double dunavant_deg2_x[] = {0.5, 0, 0.5};
constexpr double dunavant_deg2_y[] = {0, 0.5, 0.5};
constexpr double dunavant_deg2_w[] = {1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0};

// ---------- Degree 3 (4 points) ----------
constexpr double dunavant_deg3_x[] = {1.0 / 3.0, 0.2, 0.6, 0.2};
constexpr double dunavant_deg3_y[] = {1.0 / 3.0, 0.2, 0.2, 0.6};
constexpr double dunavant_deg3_w[] = {-9.0 / 32.0, 25.0 / 96.0, 25.0 / 96.0, 25.0 / 96.0};

// ---------- Degree 4 (6 points) ----------
constexpr double dunavant_deg4_x[] = {
    0.445948490915965, 0.108103018168070, 0.445948490915965, 0.091576213509771, 0.816847572980459, 0.091576213509771};
constexpr double dunavant_deg4_y[] = {
    0.445948490915965, 0.445948490915965, 0.108103018168070, 0.091576213509771, 0.091576213509771, 0.816847572980459};
constexpr double dunavant_deg4_w[] = {
    0.111690794839005, 0.111690794839005, 0.111690794839005, 0.054975871827661, 0.054975871827661, 0.054975871827661};

// ---------- Degree 5 (7 points) ----------
constexpr double dunavant_deg5_x[] = {
    1.0 / 3.0,
    0.059715871789770,
    0.470142064105115,
    0.470142064105115,
    0.797426985353087,
    0.101286507323456,
    0.101286507323456};

constexpr double dunavant_deg5_y[] = {
    1.0 / 3.0,
    0.470142064105115,
    0.059715871789770,
    0.470142064105115,
    0.101286507323456,
    0.797426985353087,
    0.101286507323456};

constexpr double dunavant_deg5_w[] = {
    0.1125,
    0.066197076394253,
    0.066197076394253,
    0.066197076394253,
    0.062969590272413,
    0.062969590272413,
    0.062969590272413};

// ---------- Degree 6 (12 points) ----------
constexpr double dunavant_deg6_x[] = {
    0.249286745170910,
    0.249286745170910,
    0.501426509658179,
    0.063089014491502,
    0.063089014491502,
    0.873821971016996,
    0.310352451033785,
    0.636502499121399,
    0.053145049844816,
    0.636502499121399,
    0.053145049844816,
    0.310352451033785};

constexpr double dunavant_deg6_y[] = {
    0.249286745170910,
    0.501426509658179,
    0.249286745170910,
    0.063089014491502,
    0.873821971016996,
    0.063089014491502,
    0.636502499121399,
    0.053145049844816,
    0.310352451033785,
    0.310352451033785,
    0.636502499121399,
    0.053145049844816};

constexpr double dunavant_deg6_w[] = {
    0.058393137863189,
    0.058393137863189,
    0.058393137863189,
    0.025422453185103,
    0.025422453185103,
    0.025422453185103,
    0.041425537809187,
    0.041425537809187,
    0.041425537809187,
    0.041425537809187,
    0.041425537809187,
    0.041425537809187};

constexpr std::array<DunavantRule<double>, 6> dunavant_rules = {{
    {1, 1, dunavant_deg1_x, dunavant_deg1_y, dunavant_deg1_w},
    {2, 3, dunavant_deg2_x, dunavant_deg2_y, dunavant_deg2_w},
    {3, 4, dunavant_deg3_x, dunavant_deg3_y, dunavant_deg3_w},
    {4, 6, dunavant_deg4_x, dunavant_deg4_y, dunavant_deg4_w},
    {5, 7, dunavant_deg5_x, dunavant_deg5_y, dunavant_deg5_w},
    {6, 12, dunavant_deg6_x, dunavant_deg6_y, dunavant_deg6_w},
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
