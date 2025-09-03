#include "test_distributed_operator.hpp"    // for test_distributed_operator
#include <htool/testing/generator_test.hpp> // for GeneratorTestComplex
#include <initializer_list>                 // for initializer_list
#include <iostream>                         // for basic_ostream, char_traits
#include <mpi.h>                            // for MPI_Finalize, MPI_Init

using namespace std;
using namespace htool;

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    bool is_error                         = false;
    const int number_of_rows              = 200;
    const int number_of_rows_increased    = 400;
    const int number_of_columns           = 200;
    const int number_of_columns_increased = 400;

    for (auto number_of_right_hand_side : {1, 5}) {
        for (auto offdiagonal_approximation : {true, false}) {
            for (auto use_buffer : {true, false}) {
                for (auto operation : {'N', 'T'}) {
                    for (auto data_type : {DataType::Matrix, DataType::HMatrix}) {
                        std::vector<double> tolerances{1e-14};
                        if (data_type == DataType::HMatrix) {
                            tolerances.push_back(1e-3);
                        }
                        for (auto epsilon : tolerances) {
                            std::cout << use_buffer << " " << epsilon << " " << number_of_right_hand_side << " " << operation << " " << epsilon << " " << offdiagonal_approximation << "\n";

                            // Square matrix
                            is_error = is_error || test_distributed_operator<double, GeneratorTestDoubleSymmetric>(number_of_rows, number_of_columns, number_of_right_hand_side, use_buffer, 'N', 'N', operation, offdiagonal_approximation, data_type, epsilon);
                            is_error = is_error || test_distributed_operator<double, GeneratorTestDoubleSymmetric>(number_of_rows, number_of_columns, number_of_right_hand_side, use_buffer, 'S', 'L', operation, offdiagonal_approximation, data_type, epsilon);
                            is_error = is_error || test_distributed_operator<double, GeneratorTestDoubleSymmetric>(number_of_rows, number_of_columns, number_of_right_hand_side, use_buffer, 'S', 'U', operation, offdiagonal_approximation, data_type, epsilon);

                            // // Rectangular matrix
                            is_error = is_error || test_distributed_operator<double, GeneratorTestDouble>(number_of_rows_increased, number_of_columns, number_of_right_hand_side, use_buffer, 'N', 'N', operation, offdiagonal_approximation, data_type, epsilon);
                            is_error = is_error || test_distributed_operator<double, GeneratorTestDouble>(number_of_rows, number_of_columns_increased, number_of_right_hand_side, use_buffer, 'N', 'N', operation, offdiagonal_approximation, data_type, epsilon);
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
