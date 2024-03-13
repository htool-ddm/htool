#include "../test_hmatrix_product.hpp"
#include <htool/hmatrix/hmatrix.hpp>

using namespace std;
using namespace htool;

int main(int argc, char *argv[]) {

    MPI_Init(&argc, &argv);
    bool is_error          = false;
    const int n1           = 400;
    const int n1_increased = 600;
    const int n2           = 400;
    const int n2_increased = 600;
    const double margin    = 10;

    for (auto use_local_cluster : {true, false}) {
        for (auto epsilon : {1e-14, 1e-6}) {
            for (auto n3 : {100}) {
                for (auto operation : {'N', 'T'}) {
                    // Square matrix
                    is_error = is_error || test_hmatrix_product<std::complex<double>, GeneratorTestComplexSymmetric>(operation, 'N', n1, n2, n3, 'N', 'N', 'N', use_local_cluster, epsilon, margin);
                    is_error = is_error || test_hmatrix_product<std::complex<double>, GeneratorTestComplexSymmetric>(operation, 'N', n1, n2, n3, 'L', 'S', 'L', use_local_cluster, epsilon, margin);
                    is_error = is_error || test_hmatrix_product<std::complex<double>, GeneratorTestComplexSymmetric>(operation, 'N', n1, n2, n3, 'L', 'S', 'U', use_local_cluster, epsilon, margin);

                    // Rectangle matrix
                    is_error = is_error || test_hmatrix_product<std::complex<double>, GeneratorTestComplex>(operation, 'N', n1_increased, n2, n3, 'N', 'N', 'N', use_local_cluster, epsilon, margin);
                    is_error = is_error || test_hmatrix_product<std::complex<double>, GeneratorTestComplex>(operation, 'N', n1, n2_increased, n3, 'N', 'N', 'N', use_local_cluster, epsilon, margin);
                }

                // TODO: fix 'C' operation, missing some conj operations somewhere
                // for (auto operation : {'N', 'C'}) {

                //     // Square matrix
                //     is_error = is_error || test_hmatrix_product<std::complex<double>, GeneratorTestComplexSymmetric>(operation, 'N', n1, n2, n3, 'N', 'N', 'N', use_local_cluster, epsilon, margin);
                //     is_error = is_error || test_hmatrix_product<std::complex<double>, GeneratorTestComplexHermitian>(operation, 'N', n1, n2, n3, 'L', 'H', 'L', use_local_cluster, epsilon, margin);
                //     is_error = is_error || test_hmatrix_product<std::complex<double>, GeneratorTestComplexHermitian>(operation, 'N', n1, n2, n3, 'L', 'H', 'U', use_local_cluster, epsilon, margin);

                //     // Rectangle matrix
                //     is_error = is_error || test_hmatrix_product<std::complex<double>, GeneratorTestComplex>(operation, 'N', n1_increased, n2, n3, 'N', 'N', 'N', use_local_cluster, epsilon, margin);
                //     is_error = is_error || test_hmatrix_product<std::complex<double>, GeneratorTestComplex>(operation, 'N', n1, n2_increased, n3, 'N', 'N', 'N', use_local_cluster, epsilon, margin);
                // }
            }
        }
    }
    MPI_Finalize();
    if (is_error) {
        return 1;
    }
    return 0;
}
