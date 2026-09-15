#ifndef HTOOL_QUADRATURES_HPP
#define HTOOL_QUADRATURES_HPP
#include <array>
namespace htool {

template <typename T, int dim>
struct QuadPoint {
    std::array<T, dim> point;
    T w;
};

template <typename RuleArray>
const auto &find_best_rule(std::size_t requested_degree, const RuleArray &rules) {
    for (std::size_t i = 0; i < rules.size(); ++i) {
        if (rules[i].degree >= requested_degree) {
            return rules[i];
        }
    }
    return rules[rules.size() - 1];
}

} // namespace htool
#endif
