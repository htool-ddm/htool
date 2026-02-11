#include "test_quadratures.hpp"
#include "htool/quadratures/gauss_legendre.hpp"
#include "htool/quadratures/triangle.hpp"

using namespace htool;
int main() {
    bool test = 0;

    // Gauss Legendre
    for (int i = 0; i < gauss_legendre_rules<double>.size(); i++) {
        test = test || test_interval_quadratures<double>(htool::gauss_legendre_rules<double>[i].degree, htool::gauss_legendre_rules<double>[i]);
    }

    for (int i = 0; i < gauss_legendre_rules<float>.size(); i++) {
        test = test || test_interval_quadratures<float>(htool::gauss_legendre_rules<float>[i].degree, htool::gauss_legendre_rules<float>[i]);
    }

    // Triangle
    for (int i = 0; i < triangle_rules<double>.size(); i++) {
        test = test || test_triangle_quadratures<double>(triangle_rules<double>[i].degree, triangle_rules<double>[i]);
    }

    for (int i = 0; i < triangle_rules<float>.size(); i++) {
        test = test || test_triangle_quadratures<float>(triangle_rules<float>[i].degree, triangle_rules<float>[i]);
    }

    return test;
}
