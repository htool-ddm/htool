#include "../test_matrix_triangular_solve.hpp" // for test_matrix_triangula...
#include <complex>                             // for complex, operator/, abs
#include <initializer_list>                    // for initializer_list
#include <iostream>                            // for basic_ostream, operat...

using namespace std;
using namespace htool;

int main(int, char *[]) {

    bool is_error            = false;
    const int number_of_rows = 200;

    for (auto number_of_rhs : {1, 100}) {
        for (auto side : {'L', 'R'}) {
            for (auto operation : {'N', 'T', 'C'}) {
                for (auto diag : {'N', 'U'}) {
                    std::cout << number_of_rhs << " " << side << " " << operation << " " << diag << "\n";
                    // Square matrix
                    is_error = is_error || test_matrix_triangular_solve<std::complex<double>>(number_of_rows, number_of_rhs, side, operation, diag);
                }
            }
        }
    }

    if (is_error) {
        return 1;
    }
    return 0;
}
