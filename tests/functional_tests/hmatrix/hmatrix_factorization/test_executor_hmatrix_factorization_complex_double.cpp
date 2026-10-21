#include "../test_executor_hmatrix_factorization.hpp"

using namespace std;
using namespace htool;

int main(int, char *[]) {

    bool is_error       = false;
    const double margin = 10;
    const int n2        = 100;

    for (auto n1 : {10, 1000}) { // 10 to deal with trivial case
        for (auto epsilon : {1e-3, 1e-6}) {
            for (auto eta : {10., 100.}) {
                for (auto trans : {'N', 'T'}) {
                    for (int number_of_threads : {1, 4}) {
                        is_error = is_error || test_executor_hmatrix_lu<std::complex<double>, GeneratorTestComplexSymmetric>(trans, n1, n2, epsilon, eta, margin, number_of_threads);
                        is_error = is_error || test_lazy_hmatrix_lu<std::complex<double>, GeneratorTestComplexSymmetric>(trans, n1, n2, epsilon, eta, margin, number_of_threads);
                    }
                }
            }
        }
    }

    for (auto epsilon : {1e-4, 1e-6}) {
        is_error = is_error || test_lazy_hmatrix_lu_first_kind<std::complex<double>>(2000, epsilon, 10, 4);
    }

    std::cout << "+++++++++++++++++++++++++++++++++++++++++++" << '\n';
    if (is_error) {
        return 1;

    } else {
        std::cout << "SUCCESS: All test_executor_hmatrix_factorization cases passed." << '\n';
    }
    std::cout << "+++++++++++++++++++++++++++++++++++++++++++" << '\n';

    return 0;
}
