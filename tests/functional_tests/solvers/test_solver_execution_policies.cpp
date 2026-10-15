#include "test_solver_ddm.hpp"                // for test_solver_ddm
#include "test_solver_ddm_adding_overlap.hpp" // for test_solver_ddm_adding_overlap
#include "test_solver_wo_overlap.hpp"         // for test_solver_wo_overlap
#include <htool/solvers/utility.hpp>          // for DDMSolverBuilder
#include <iostream>                           // for cout
#include <mpi.h>                              // for MPI_Init
#include <string>                             // for string
#include <utility>                            // for forward

// DDMSolverBuilder factorizing and solving with its local HMatrix with ExecutionPolicy (and building it, when it does)
template <typename ExecutionPolicy, typename CoefficientPrecision, typename CoordinatePrecision>
struct DDMSolverBuilderWithExecutionPolicy : DDMSolverBuilder<CoefficientPrecision, CoordinatePrecision> {
    template <typename... Args>
    DDMSolverBuilderWithExecutionPolicy(Args &&...args) : DDMSolverBuilder<CoefficientPrecision, CoordinatePrecision>(ExecutionPolicy{}, std::forward<Args>(args)...) {}
};

template <typename ExecutionPolicy>
bool test_solver_execution_policy(int argc, char *argv[], const std::string &datapath) {
    using SolverBuilder = DDMSolverBuilderWithExecutionPolicy<ExecutionPolicy, double, double>;
    bool is_error       = false;
    for (auto symmetry : {'N', 'S'}) {
        char UPLO = symmetry == 'N' ? 'N' : 'L';
        is_error  = is_error || test_solver_wo_overlap<double, double, SolverBuilder>(argc, argv, 5, symmetry, UPLO, datapath);
        is_error  = is_error || test_solver_ddm_adding_overlap<double, double, SolverBuilder, DefaultApproximationBuilder<double, double>>(argc, argv, 5, symmetry, symmetry, UPLO, datapath);
        is_error  = is_error || test_solver_ddm<double, double, SolverBuilder>(argc, argv, 5, symmetry, symmetry, UPLO, datapath);
    }
    return is_error;
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    if (argc < 2) {
        cout << "usage: " << argv[0] << " datapath\n"; // LCOV_EXCL_LINE
        return 1;                                      // LCOV_EXCL_LINE
    }
    std::string datapath_final = std::string(argv[1]) + "/output_sym/";

    bool is_error = test_solver_execution_policy<exec_compat::parallel_policy>(argc, argv, datapath_final);
    is_error      = is_error || test_solver_execution_policy<omp_task_policy<double, double>>(argc, argv, datapath_final);

    MPI_Finalize();
    return is_error ? 1 : 0;
}
