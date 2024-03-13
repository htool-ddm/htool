#include "test_solver_ddm.hpp"
#include "test_solver_ddm_adding_overlap.hpp"
#include "test_solver_wo_overlap.hpp"

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

                is_error = is_error || test_solver_wo_overlap(argc, argv, nb_rhs, symmetry, datapath_final);
                is_error = is_error || test_solver_ddm_adding_overlap(argc, argv, nb_rhs, data_symmetry, symmetry, datapath_final);
                is_error = is_error || test_solver_ddm(argc, argv, nb_rhs, data_symmetry, symmetry, datapath_final);
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
