#include "test_quadratures.hpp"

int main() {
    bool test = 0;

    // Gauss Legendre
    for (int i = 0; i < gauss_legendre_rules.size(); i++) {
        test = test || test_gauss_legendre_quadratures(gauss_legendre_rules[i].degree, gauss_legendre_rules[i]);
    }

    // Dunavant
    for (int i = 0; i < dunavant_rules.size(); i++) {
        test = test || test_dunavant_quadratures(dunavant_rules[i].degree, dunavant_rules[i]);
    }

    return test;
}
