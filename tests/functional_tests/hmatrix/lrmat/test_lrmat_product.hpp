#include "test_lrmat_lrmat_product.hpp"
#include "test_lrmat_matrix_product.hpp"
#include "test_matrix_lrmat_product.hpp"
#include "test_matrix_matrix_product.hpp"
#include <htool/testing/generate_test_case.hpp>

using namespace std;
using namespace htool;

template <typename T, typename GeneratorTestType, class Compressor>
bool test_lrmat_product(char transa, char transb, int n1, int n2, int n3, htool::underlying_type<T> epsilon, htool::underlying_type<T> additional_compression_tolerance, std::array<htool::underlying_type<T>, 4> additional_lrmat_sum_tolerances) {
    bool is_error       = false;
    const int ndistance = 4;
    htool::underlying_type<T> distance[ndistance];
    distance[0] = 15;
    distance[1] = 20;
    distance[2] = 30;
    distance[3] = 40;
    for (int idist = 0; idist < ndistance; idist++) {
        TestCase<T, GeneratorTestType> test_case(transa, transb, n1, n2, n3, distance[idist], distance[idist] + 10, 'N', 'N', 'N');
        is_error = is_error || test_lrmat_lrmat_product<T, GeneratorTestType, Compressor>(test_case, epsilon, additional_compression_tolerance, additional_lrmat_sum_tolerances[0]);
        is_error = is_error || test_lrmat_matrix_product<T, GeneratorTestType, Compressor>(test_case, epsilon, additional_compression_tolerance, additional_lrmat_sum_tolerances[1]);
        is_error = is_error || test_matrix_lrmat_product<T, GeneratorTestType, Compressor>(test_case, epsilon, additional_compression_tolerance, additional_lrmat_sum_tolerances[2]);
        is_error = is_error || test_matrix_matrix_product<T, GeneratorTestType, Compressor>(test_case, epsilon, additional_compression_tolerance, additional_lrmat_sum_tolerances[3]);
    }

    return is_error;
}
