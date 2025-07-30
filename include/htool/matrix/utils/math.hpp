#ifndef HTOOL_MATRIX_UTILS_MATH_HPP
#define HTOOL_MATRIX_UTILS_MATH_HPP
#include "../../misc/misc.hpp"
namespace htool {

template <typename Mat>
auto normFrob(const Mat &A) {
    using T                        = typename Mat::value_type;
    htool::underlying_type<T> norm = 0;
    for (int j = 0; j < A.nb_rows(); j++) {
        for (int k = 0; k < A.nb_cols(); k++) {
            norm = norm + std::pow(std::abs(A(j, k)), 2);
        }
    }
    return sqrt(norm);
}

template <typename Mat>
std::pair<int, int> argmax(const Mat &M) {
    using T = typename Mat::value_type;
    int p   = std::max_element(M.data(), M.data() + M.nb_cols() * M.nb_rows(), [](T a, T b) { return std::abs(a) < std::abs(b); }) - M.data();
    return std::pair<int, int>(p % M.nb_rows(), p / M.nb_rows());
}

} // namespace htool
#endif
