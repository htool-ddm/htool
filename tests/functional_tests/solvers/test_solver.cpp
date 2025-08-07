#include "test_solver_ddm.hpp"                // for test_so...
#include "test_solver_ddm_adding_overlap.hpp" // for test_so...
#include "test_solver_wo_overlap.hpp"         // for test_so...
#include <algorithm>                          // for max, copy
#include <complex>                            // for complex
#include <htool/solvers/utility.hpp>          // for DDMSolv...
#include <initializer_list>                   // for initial...
#include <iostream>                           // for basic_o...
#include <mpi.h>                              // for MPI_Fin...
#include <string>                             // for operator<<
#include <vector>                             // for vector

int main(int argc, char *argv[]) {
    // Initialize the MPI environment
    MPI_Init(&argc, &argv);

    // Input file
    if (argc < 2) { // argc should be 5 or more for correct execution
        // We print argv[0] assuming it is the program name
        cout << "usage: " << argv[0] << " datapath\n"; // LCOV_EXCL_LINE
        return 1;                                      // LCOV_EXCL_LINE
    }
    string datapath = argv[1];

    bool is_error = false;

    for (auto nb_rhs : {1, 5}) {
        for (auto data_symmetry : {'N', 'S'}) {
            std::string datapath_final   = data_symmetry == 'S' ? datapath + "/output_sym/" : datapath + "/output_non_sym/";
            std::vector<char> symmetries = {'N'};
            if (data_symmetry == 'S') {
                symmetries.push_back('S');
            }
            for (auto symmetry : symmetries) {
                std::cout << nb_rhs << " " << data_symmetry << " " << symmetry << "\n";

                is_error = is_error || test_solver_wo_overlap<DDMSolverWithDenseLocalSolver<complex<double>, double>>(argc, argv, nb_rhs, symmetry, symmetry == 'N' ? 'N' : 'L', datapath_final);
                is_error = is_error || test_solver_ddm_adding_overlap<DDMSolverWithDenseLocalSolver<std::complex<double>>>(argc, argv, nb_rhs, data_symmetry, symmetry, datapath_final);
                is_error = is_error || test_solver_ddm<DDMSolverWithDenseLocalSolver<complex<double>>>(argc, argv, nb_rhs, data_symmetry, symmetry, symmetry == 'N' ? 'N' : 'L', datapath_final);

                std::vector<char> storage;
                if (symmetry == 'N') {
                    storage.push_back('N');
                } else {
                    storage.push_back('U');
                    storage.push_back('L');
                }
                for (auto UPLO : storage) {
                    is_error = is_error || test_solver_wo_overlap<DDMSolverBuilder<complex<double>, double>>(argc, argv, nb_rhs, symmetry, UPLO, datapath_final);
                    test_solver_ddm_adding_overlap(argc, argv, nb_rhs, data_symmetry, symmetry, UPLO, datapath_final);
                    is_error = is_error || test_solver_ddm<DDMSolverBuilder<complex<double>>>(argc, argv, nb_rhs, data_symmetry, symmetry, UPLO, datapath_final);
                }
            }
        }
    }

    // Finalize the MPI environment.
    MPI_Finalize();
    if (is_error) {
        return 1;
    }
    return 0;
}
