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

    bool is_error              = false;
    std::string datapath_final = datapath + "/output_non_sym/";

    for (auto nb_rhs : {1, 5}) {
        for (auto data_symmetry : {'N'}) {
            std::vector<char> symmetries = {'N'};
            for (auto symmetry : symmetries) {
                std::cout << nb_rhs << " " << data_symmetry << " " << symmetry << "\n";

                is_error = is_error || test_solver_wo_overlap<std::complex<double>, double, DDMSolverWithDenseLocalSolver<std::complex<double>, double>>(argc, argv, nb_rhs, symmetry, symmetry == 'N' ? 'N' : 'L', datapath_final);
                is_error = is_error || test_solver_ddm_adding_overlap<std::complex<double>, double, DDMSolverWithDenseLocalSolver<std::complex<double>>>(argc, argv, nb_rhs, data_symmetry, symmetry, symmetry == 'N' ? 'N' : 'L', datapath_final);
                is_error = is_error || test_solver_ddm<std::complex<double>, double, DDMSolverWithDenseLocalSolver<complex<double>>>(argc, argv, nb_rhs, data_symmetry, symmetry, symmetry == 'N' ? 'N' : 'L', datapath_final);

                is_error = is_error || test_solver_wo_overlap<std::complex<double>, double, DDMSolverBuilder<std::complex<double>, double>>(argc, argv, nb_rhs, symmetry, 'N', datapath_final);
                test_solver_ddm_adding_overlap<std::complex<double>, double, DDMSolverBuilder<std::complex<double>, double>>(argc, argv, nb_rhs, data_symmetry, symmetry, 'N', datapath_final);
                is_error = is_error || test_solver_ddm<std::complex<double>, double, DDMSolverBuilder<complex<double>>>(argc, argv, nb_rhs, data_symmetry, symmetry, 'N', datapath_final);
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
