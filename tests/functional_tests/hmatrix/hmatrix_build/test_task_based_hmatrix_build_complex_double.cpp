#include "../test_hmatrix_build.hpp"        // for test_hmatrix_build
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
            for (auto use_local_cluster : {true, false}) {
                for (auto epsilon : {1e-14, 1e-6}) {
                    for (auto use_dense_block_generator : {true, false}) {
                        for (auto block_tree_consistency : {true, false}) {
                            std::cout << nr << " " << nc << " " << use_local_cluster << " " << epsilon << " " << use_dense_block_generator << " " << block_tree_consistency << "\n";

                            is_error = is_error || test_hmatrix_build<htool::omp_task_policy<std::complex<double>> &&, std::complex<double>, GeneratorTestComplexSymmetric>(omp_task_policy<std::complex<double>>{}, nr, nc, use_local_cluster, 'N', 'N', epsilon, use_dense_block_generator, block_tree_consistency);
                            if (nr == nc && block_tree_consistency) {
                                for (auto UPLO : {'U', 'L'}) {
                                    is_error = is_error || test_hmatrix_build<htool::omp_task_policy<std::complex<double>> &&, std::complex<double>, GeneratorTestComplexSymmetric>(omp_task_policy<std::complex<double>>{}, nr, nr, use_local_cluster, 'S', UPLO, epsilon, use_dense_block_generator, block_tree_consistency);

                                    is_error = is_error || test_hmatrix_build<htool::omp_task_policy<std::complex<double>> &&, std::complex<double>, GeneratorTestComplexHermitian>(omp_task_policy<std::complex<double>>{}, nr, nr, use_local_cluster, 'H', UPLO, epsilon, use_dense_block_generator, block_tree_consistency);
                                }
                            }
                        }
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
