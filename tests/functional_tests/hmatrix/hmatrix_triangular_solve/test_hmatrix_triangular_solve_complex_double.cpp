#include "../test_hmatrix_triangular_solve.hpp" // for test_hm...
#include <complex>                              // for complex
#include <htool/testing/generator_test.hpp>     // for Generat...
#include <initializer_list>                     // for initial...
#include <iostream>                             // for basic_o...

using namespace std;
using namespace htool;

int main(int, char *[]) {

    bool is_error       = false;
    const int n1        = 400;
    const double margin = 1;

    for (auto number_of_rhs : {100}) {
        for (auto epsilon : {1e-6, 1e-10}) {
            for (auto side : {'L', 'R'}) {
                for (auto operation : {'N', 'T', 'C'}) {
                    std::cout << epsilon << " " << number_of_rhs << " " << side << " " << operation << "\n";
                    is_error = is_error || test_hmatrix_triangular_solve<std::complex<double>, GeneratorTestComplexHermitian>(side, operation, n1, number_of_rhs, epsilon, margin);
                }
            }
        }
    }
    if (is_error) {
        return 1;
    }
    return 0;
}
