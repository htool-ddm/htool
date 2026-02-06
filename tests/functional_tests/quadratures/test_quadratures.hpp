

// #include "htool/hmatrix/lrmat/dunavant_quadratures.hpp"
#include "htool/quadratures/dunavant.hpp"
#include "htool/quadratures/gauss_legendre.hpp"
#include <cmath>
#include <iostream>

using namespace htool;

inline bool test_gauss_legendre_quadratures(int max_polynomial_degree, GaussLegendreRule<double> rule) {

    bool is_error = 0;
    double error;

    for (int p = 0; p <= max_polynomial_degree; p++) {
        double sum       = 0;
        double reference = 1. / (p + 1) - std::pow(-1, p + 1) / (p + 1);
        for (int i = 0; i < rule.nb_points; ++i) {
            sum += rule.w[i] * std::pow(rule.x[i], p);
        }

        error    = std::abs(sum - reference);
        is_error = is_error || !(error < 1e-10);
        std::cout << "Gauss Legendre (degree,polynome order, sum-reference=error, is_error): " << rule.degree << ", " << p << ", " << sum << "-" << reference << "=" << error << ", " << is_error << "\n";
        if (is_error == true) {
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

inline bool test_dunavant_quadratures(int max_polynomial_degree, DunavantRule<double> rule) {
    bool is_error = false;
    double error;

    for (int p = 0; p <= max_polynomial_degree; p++) {
        for (int q = 0; q <= max_polynomial_degree - p; q++) {
            double sum       = 0;
            double reference = static_cast<double>(factorial(p) * factorial(q)) / factorial(p + q + 2);
            for (int i = 0; i < rule.nb_points; ++i) {
                sum += rule.w[i] * std::pow(rule.x[i], p) * std::pow(rule.y[i], q);
            }

            error    = std::abs(sum - reference);
            is_error = is_error || !(error < 1e-10);
            std::cout << "Dunavant (degree,polynome order, sum-reference=error, is_error): " << rule.degree << ", " << p + q << ", " << sum << "-" << reference << "=" << error << ", " << is_error << "\n";
            if (is_error == true) {
                return is_error;
            }
        }
    }
    return is_error;
}
