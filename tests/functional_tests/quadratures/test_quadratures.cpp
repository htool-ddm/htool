#include "test_quadratures.hpp"
#include "htool/quadratures/dunavant.hpp"
#include "htool/quadratures/gauss_legendre.hpp"
#include "htool/quadratures/radon_laurie.hpp"

using namespace htool;
int main() {
    bool test = 0;

    // Gauss Legendre
    for (int i = 0; i < htool::gauss_legendre_rules<double>.size(); i++) {
        test = test || test_interval_quadratures<double>(htool::gauss_legendre_rules<double>[i].degree, htool::gauss_legendre_rules<double>[i]);
    }

    for (int i = 0; i < htool::gauss_legendre_rules<float>.size(); i++) {
        test = test || test_interval_quadratures<float>(htool::gauss_legendre_rules<float>[i].degree, htool::gauss_legendre_rules<float>[i]);
    }

    // // Dunavant
    // for (int i = 0; i < dunavant_rules<double>.size(); i++) {
    //     test = test || test_triangle_quadratures<double>(dunavant_rules<double>[i].degree, dunavant_rules<double>[i]);
    // }

    // for (int i = 0; i < dunavant_rules<float>.size(); i++) {
    //     test = test || test_triangle_quadratures<float>(dunavant_rules<float>[i].degree, dunavant_rules<float>[i]);
    // }

    // Laurie
    for (int i = 0; i < laurie_rules<float>.size(); i++) {
        test = test || test_triangle_quadratures<float>(laurie_rules<float>[i].degree, laurie_rules<float>[i]);
    }

    for (int i = 0; i < laurie_rules<double>.size(); i++) {
        test = test || test_triangle_quadratures<double>(laurie_rules<double>[i].degree, laurie_rules<double>[i]);
    }

    // Radon
    for (int i = 0; i < radon_rules<float>.size(); i++) {
        test = test || test_triangle_quadratures<float, RadonRule<float>>(radon_rules<float>[i].degree, radon_rules<float>[i]);
    }

    for (int i = 0; i < radon_rules<double>.size(); i++) {
        test = test || test_triangle_quadratures<double, RadonRule<double>>(radon_rules<double>[i].degree, radon_rules<double>[i]);
    }

    return test;
}
