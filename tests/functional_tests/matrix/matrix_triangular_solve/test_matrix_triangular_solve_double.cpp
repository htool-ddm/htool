#include "../test_matrix_triangular_solve.hpp"

using namespace std;
using namespace htool;

int main(int, char *[]) {

    bool is_error            = false;
    const int number_of_rows = 200;

    for (auto number_of_rhs : {1, 100}) {
        for (auto side : {'R'}) {
            for (auto operation : {'N'}) {
                std::cout << number_of_rhs << " " << side << " " << operation << "\n";
                // Square matrix
                is_error = is_error || test_matrix_triangular_solve<double>(number_of_rows, number_of_rhs, side, operation);
            }
            is_error = is_error || test_symmetric_matrix_triangular_solve<double>(number_of_rows, number_of_rhs, side);
        }
    }

    if (is_error) {
        return 1;
    }
    return 0;
}
