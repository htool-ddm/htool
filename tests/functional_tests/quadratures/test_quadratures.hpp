

// #include "htool/hmatrix/lrmat/dunavant_quadratures.hpp"
#include <cmath>
#include <iostream>
#include <limits>

template <typename T>
constexpr T ipow(T base, int exp) {
    T result = 1;
    for (int i = 0; i < exp; ++i)
        result *= base;
    return result;
}

template <typename T, typename QuadRule>
bool test_interval_quadratures(int max_polynomial_degree, QuadRule rule) {

    bool is_error = 0;
    T error;

    for (int p = 0; p <= max_polynomial_degree; p++) {
        T sum       = 0;
        T reference = 1. / (p + 1) - ipow(-1., p + 1) / (p + 1);
        for (int i = 0; i < rule.nb_points; ++i) {
            sum += rule.quad_points[i].w * ipow(rule.quad_points[i].point[0], p);
        }

        error    = std::abs(sum - reference);
        is_error = is_error || !(error < std::numeric_limits<T>::epsilon() * 10);
        if (is_error == true) {
            std::cout << rule.name << " (degree,polynome order, sum-reference=error, is_error): " << rule.degree << ", " << p << ", " << sum << "-" << reference << "=" << error << ", " << is_error << "\n";
            return is_error;
        }
    }
    return is_error;
}

constexpr int factorial(int n) {
    int r = 1;
    for (int i = 2; i <= n; ++i)
        r *= i;
    return r;
}

// inline bool test_dunavant_quadratures(int max_polynomial_degree, DunavantRule<double> rule) {
//     bool is_error = false;
//     double error;

//     for (int p = 0; p <= max_polynomial_degree; p++) {
//         for (int q = 0; q <= max_polynomial_degree - p; q++) {
//             double sum       = 0;
//             double reference = static_cast<double>(factorial(p) * factorial(q)) / factorial(p + q + 2);
//             for (int i = 0; i < rule.nb_points; ++i) {
//                 sum += rule.w[i] * std::pow(rule.x[i], p) * std::pow(rule.y[i], q);
//             }

//             error    = std::abs(sum - reference) / std::abs(reference);
//             is_error = is_error || !(error < std::numeric_limits<double>::epsilon() * 10);
//             std::cout << "Dunavant (degree,polynome order, sum-reference=error, is_error): " << rule.degree << ", " << p + q << ", " << sum << "-" << reference << "=" << error << ", " << is_error << "\n";
//             if (is_error == true) {
//                 return is_error;
//             }
//         }
//     }
//     return is_error;
// }

template <typename T, typename QuadRule>
bool test_triangle_quadratures(int max_polynomial_degree, QuadRule rule) {
    bool is_error = false;
    T error;

    for (int p = 0; p <= max_polynomial_degree; p++) {
        for (int q = 0; q <= max_polynomial_degree - p; q++) {
            T sum       = 0;
            T reference = static_cast<T>(factorial(p) * factorial(q)) / factorial(p + q + 2);
            for (int i = 0; i < rule.nb_points; ++i) {
                sum += rule.quad_points[i].w * ipow(rule.quad_points[i].point[0], p) * ipow(rule.quad_points[i].point[1], q);
            }

            error    = std::abs(sum - reference) / std::abs(reference);
            is_error = is_error || !(error < std::numeric_limits<T>::epsilon() * 10);
            if (is_error == true) {
                std::cout << rule.name << " (degree,polynome order, sum-reference=error, is_error): " << rule.degree << ", " << p + q << ", " << sum << "-" << reference << "=" << error << ", " << is_error << "\n";
                return is_error;
            }
        }
    }
    return is_error;
}
