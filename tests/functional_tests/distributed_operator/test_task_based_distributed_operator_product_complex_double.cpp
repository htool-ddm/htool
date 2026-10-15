#include "test_distributed_operator.hpp"    // for test_distributed_operator
#include <complex>                          // for complex, abs, operator-
#include <htool/testing/generator_test.hpp> // for GeneratorTestComplex
#include <initializer_list>                 // for initializer_list
#include <iostream>                         // for basic_ostream, char_traits
#include <mpi.h>                            // for MPI_Finalize, MPI_Init

using namespace std;
using namespace htool;

// Checks of DefaultApproximationBuilder with an omp_task_policy, run without leak detection (see the CMakeLists.txt)
int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    bool is_error                         = false;
    const int number_of_rows              = 200;
    const int number_of_rows_increased    = 400;
    const int number_of_columns           = 200;
    const int number_of_columns_increased = 400;

    for (auto number_of_right_hand_side : {1, 5}) {
        for (auto offdiagonal_approximation : {false}) {
            for (auto use_buffer : {true, false}) {
                for (auto data_type : {DataType::HMatrix}) {
                    std::vector<double> tolerances{1e-14};
                    if (data_type == DataType::HMatrix) {
                        tolerances.push_back(1e-3);
                    }
                    for (auto epsilon : tolerances) {
                        for (auto operation : {'N', 'T', 'C'}) {
                            std::cout << use_buffer << " " << epsilon << " " << number_of_right_hand_side << " " << operation << " " << epsilon << "\n";

                            // Square matrix
                            is_error = is_error || test_distributed_operator<std::complex<double>, GeneratorTestComplex>(number_of_rows, number_of_columns, number_of_right_hand_side, use_buffer, 'N', 'N', operation, offdiagonal_approximation, data_type, epsilon, true);
                            // Rectangular matrix
                            is_error = is_error || test_distributed_operator<std::complex<double>, GeneratorTestComplex>(number_of_rows_increased, number_of_columns, number_of_right_hand_side, use_buffer, 'N', 'N', operation, offdiagonal_approximation, data_type, epsilon, true);
                            is_error = is_error || test_distributed_operator<std::complex<double>, GeneratorTestComplex>(number_of_rows, number_of_columns_increased, number_of_right_hand_side, use_buffer, 'N', 'N', operation, offdiagonal_approximation, data_type, epsilon, true);
                        }
                        for (auto operation : {'N', 'T'}) {
                            std::cout << use_buffer << " " << epsilon << " " << number_of_right_hand_side << " " << operation << " " << epsilon << "\n";
                            is_error = is_error || test_distributed_operator<std::complex<double>, GeneratorTestComplexSymmetric>(number_of_rows, number_of_columns, number_of_right_hand_side, use_buffer, 'S', 'L', operation, offdiagonal_approximation, data_type, epsilon, true);
                            is_error = is_error || test_distributed_operator<std::complex<double>, GeneratorTestComplexSymmetric>(number_of_rows, number_of_columns, number_of_right_hand_side, use_buffer, 'S', 'U', operation, offdiagonal_approximation, data_type, epsilon, true);
                        }
                        for (auto operation : {'N', 'C'}) {
                            std::cout << use_buffer << " " << epsilon << " " << number_of_right_hand_side << " " << operation << " " << epsilon << "\n";
                            is_error = is_error || test_distributed_operator<std::complex<double>, GeneratorTestComplexHermitian>(number_of_rows, number_of_columns, number_of_right_hand_side, use_buffer, 'H', 'L', operation, offdiagonal_approximation, data_type, epsilon, true);
                            is_error = is_error || test_distributed_operator<std::complex<double>, GeneratorTestComplexHermitian>(number_of_rows, number_of_columns, number_of_right_hand_side, use_buffer, 'H', 'U', operation, offdiagonal_approximation, data_type, epsilon, true);
                        }
                    }
                }
            }
            // for (auto operation : {'N', 'C'}) {
            //     for (auto data_type : {DataType::HMatrix}) {
            //         std::vector<double> tolerances{1e-14};
            //         if (data_type == DataType::HMatrix) {
            //             tolerances.push_back(1e-6);
            //         }
            //         for (auto epsilon : tolerances) {
            //             std::cout < << " " << epsilon << " " << number_of_right_hand_side,use_buffer << " " << operation << "\n";
            //             // Square matrix
            //             is_error = is_error || test_distributed_operator<std::complex<double>, GeneratorTestComplex>(number_of_rows, number_of_columns, number_of_right_hand_side,use_buffer, 'N', 'N', operation, offdiagonal_approximation, data_type, epsilon, true);
            //             is_error = is_error || test_distributed_operator<std::complex<double>, GeneratorTestComplexHermitian>(number_of_rows, number_of_columns, number_of_right_hand_side,use_buffer, 'H', 'L', operation, offdiagonal_approximation, data_type, epsilon, true);
            //             is_error = is_error || test_distributed_operator<std::complex<double>, GeneratorTestComplexHermitian>(number_of_rows, number_of_columns, number_of_right_hand_side,use_buffer, 'H', 'U', operation, offdiagonal_approximation, data_type, epsilon, true);

            //             // Rectangular matrix
            //             is_error = is_error || test_distributed_operator<std::complex<double>, GeneratorTestComplex>(number_of_rows_increased, number_of_columns, number_of_right_hand_side,use_buffer, 'N', 'N', operation, offdiagonal_approximation, data_type, epsilon, true);
            //             is_error = is_error || test_distributed_operator<std::complex<double>, GeneratorTestComplex>(number_of_rows, number_of_columns_increased, number_of_right_hand_side,use_buffer, 'N', 'N', operation, offdiagonal_approximation, data_type, epsilon, true);
            //         }
            //     }
            // }
        }
    }

    MPI_Finalize();

    if (is_error) {
        return 1;
    }
    return 0;
}
