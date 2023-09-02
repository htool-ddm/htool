#include "test_distributed_operator.hpp"

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
            for (auto use_permutation : {true, false}) {
                for (auto operation : {'N', 'T'}) {
                    for (auto data_type : {DataType::Matrix, DataType::HMatrix, DataType::DefaultHMatrix}) {
                        std::vector<double> tolerances{1e-14};
                        if (data_type == DataType::HMatrix || data_type == DataType::DefaultHMatrix) {
                            tolerances.push_back(1e-3);
                        }
                        for (auto epsilon : tolerances) {
                            std::cout << use_permutation << " " << epsilon << " " << number_of_right_hand_side << " " << operation << " " << epsilon << "\n";

                            // Square matrix
                            is_error = is_error || test_distributed_operator<double, GeneratorTestDoubleSymmetric>(number_of_rows, number_of_columns, number_of_right_hand_side, use_permutation, 'N', 'N', operation, offdiagonal_approximation, data_type, epsilon);
                            is_error = is_error || test_distributed_operator<double, GeneratorTestDoubleSymmetric>(number_of_rows, number_of_columns, number_of_right_hand_side, use_permutation, 'S', 'L', operation, offdiagonal_approximation, data_type, epsilon);
                            is_error = is_error || test_distributed_operator<double, GeneratorTestDoubleSymmetric>(number_of_rows, number_of_columns, number_of_right_hand_side, use_permutation, 'S', 'U', operation, offdiagonal_approximation, data_type, epsilon);

                            // Rectangular matrix
                            is_error = is_error || test_distributed_operator<double, GeneratorTestDouble>(number_of_rows_increased, number_of_columns, number_of_right_hand_side, use_permutation, 'N', 'N', operation, offdiagonal_approximation, data_type, epsilon);
                            is_error = is_error || test_distributed_operator<double, GeneratorTestDouble>(number_of_rows, number_of_columns_increased, number_of_right_hand_side, use_permutation, 'N', 'N', operation, offdiagonal_approximation, data_type, epsilon);
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
