#include "../test_lrmat_lrmat_addition.hpp"
#include <htool/hmatrix/lrmat/partialACA.hpp>
#include <htool/testing/generator_input.hpp>
#include <htool/testing/generator_test.hpp>
int main(int, char const *[]) {
    bool is_error          = false;
    const int greater_size = 400;
    const int lower_size   = 200;
    const int dummy        = 100;

    is_error = is_error || test_lrmat_lrmat_addition<double, GeneratorTestDouble, partialACA<double>>('N', 'N', greater_size, lower_size, dummy, 1e-10);
    is_error = is_error || test_lrmat_lrmat_addition<double, GeneratorTestDouble, partialACA<double>>('N', 'N', lower_size, greater_size, dummy, 1e-10);
    is_error = is_error || test_lrmat_lrmat_addition<double, GeneratorTestDouble, partialACA<double>>('N', 'N', greater_size, lower_size, dummy, 1e-6);
    is_error = is_error || test_lrmat_lrmat_addition<double, GeneratorTestDouble, partialACA<double>>('N', 'N', lower_size, greater_size, dummy, 1e-6);
    if (is_error) {
        return 1;
    }

    return 0;
}
