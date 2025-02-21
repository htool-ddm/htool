#include "../test_hmatrix_builder.hpp"      // for test_hmatrix_build
#include <algorithm>                        // for max
#include <complex>                          // for complex, abs, operator-
#include <htool/testing/generator_test.hpp> // for GeneratorTestComplexSymm...
#include <initializer_list>                 // for initializer_list
#include <iostream>                         // for basic_ostream, char_traits
#include <mpi.h>                            // for MPI_Finalize, MPI_Init

using namespace std;
using namespace htool;

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    bool is_error = false;

    for (auto nr : {200, 400}) {
        for (auto nc : {200, 400}) {
            for (auto epsilon : {1e-14, 1e-6}) {
                std::cout << nr << " " << nc << " " << epsilon << "\n";

                is_error = is_error || test_hmatrix_builder<std::complex<double>, GeneratorTestComplexSymmetric>(nr, nc, 'N', 'N', epsilon);
                if (nr == nc) {
                    for (auto UPLO : {'U', 'L'}) {
                        is_error = is_error || test_hmatrix_builder<std::complex<double>, GeneratorTestComplexSymmetric>(nr, nr, 'S', UPLO, epsilon);

                        is_error = is_error || test_hmatrix_builder<std::complex<double>, GeneratorTestComplexHermitian>(nr, nr, 'H', UPLO, epsilon);
                    }
                }
            }
        }
    }

    MPI_Finalize();

    if (is_error) {
        return 1;
    }
    return 0;
}
