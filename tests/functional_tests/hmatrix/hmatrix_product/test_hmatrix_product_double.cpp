#include "../test_hmatrix_product.hpp"      // for test_hm...
#include <algorithm>                        // for max, copy
#include <array>                            // for array
#include <htool/testing/generator_test.hpp> // for Generat...
#include <initializer_list>                 // for initial...
#include <iostream>                         // for basic_o...
#include <mpi.h>                            // for MPI_Fin...

using namespace std;
using namespace htool;

int main(int argc, char *argv[]) {

    MPI_Init(&argc, &argv);
    bool is_error = false;
    const std::array<double, 5> additional_lrmat_sum_tolerances{10., 10., 10., 10., 10.};

    for (auto epsilon : {1e-6, 1e-10}) {
        for (auto n1 : {200, 400}) {
            for (auto n3 : {100}) {
                for (auto use_local_cluster : {true, false}) {
                    for (auto n2 : {200, 400}) {
                        for (auto transa : {'N', 'T'}) {
                            for (auto transb : {'N', 'T'}) {
                                std::cout << "hmatrix product: " << use_local_cluster << " " << epsilon << " " << n1 << " " << n2 << " " << n3 << " " << transa << " " << transb << "\n";

                                is_error = is_error || test_hmatrix_product<double, GeneratorTestDouble>(transa, transb, n1, n2, n3, use_local_cluster, epsilon, additional_lrmat_sum_tolerances);
                            }
                        }
                    }
                }
                for (auto side : {'L', 'R'}) {
                    for (auto UPLO : {'U', 'L'}) {
                        const double margin = 0;
                        std::cout << "symmetric matrix product: " << n1 << " " << n3 << " " << side << " " << UPLO << " " << epsilon << "\n";
                        is_error = is_error || test_symmetric_hmatrix_product<double, GeneratorTestDoubleSymmetric>(n1, n3, side, UPLO, epsilon, margin);
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
