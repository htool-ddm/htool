#include "../test_matrix_product.hpp" // for test_matrix_product, test_matr...
#include <initializer_list>           // for initializer_list
#include <iostream>                   // for basic_ostream, operator<<, cha...

using namespace std;
using namespace htool;

int main(int, char *[]) {

    bool is_error = false;
    for (auto n1 : {200, 400}) {
        for (auto n3 : {1, 5}) {
            for (auto n2 : {200, 400}) {
                for (auto transa : {'N', 'T'}) {
                    for (auto transb : {'N', 'T'}) {
                        std::cout << "matrix product: " << n1 << " " << n2 << " " << n3 << " " << transa << " " << transb << "\n";
                        // Square matrix
                        is_error = is_error || test_matrix_product<double>(n1, n2, n3, transa, transb);
                    }
                }
            }
            for (auto side : {'L', 'R'}) {
                for (auto UPLO : {'U', 'L'}) {
                    std::cout << "symmetric matrix product: " << n1 << " " << n3 << " " << side << " " << UPLO << "\n";
                    is_error = is_error || test_matrix_symmetric_product<double>(n1, n3, side, UPLO);
                }
            }
        }
    }

    if (is_error) {
        return 1;
    }
    return 0;
}
