#include <algorithm>                              // for max
#include <complex>                                // for complex
#include <htool/basic_types/vector.hpp>           // for bytes_to_vector
#include <htool/clustering/cluster_node.hpp>      // for Cluster
#include <htool/clustering/cluster_output.hpp>    // for read_cluster_tree
#include <htool/distributed_operator/utility.hpp> // for DefaultApproximati...
#include <htool/hmatrix/hmatrix.hpp>              // for HMatrix
#include <htool/hmatrix/hmatrix_output.hpp>       // for print_hmatrix_info...
#include <htool/matrix/matrix.hpp>                // for Matrix, normFrob
#include <htool/misc/misc.hpp>                    // for underlying_type
#include <htool/misc/user.hpp>                    // for NbrToStr
#include <htool/testing/generator_test.hpp>       // for GeneratorInUserNum...
#include <htool/testing/point.hpp>                // for Cplx
#include <initializer_list>                       // for initializer_list
#include <iostream>                               // for basic_ostream, ope...
#include <memory>                                 // for allocator, __share...
#include <mpi.h>                                  // for MPI_COMM_WORLD
#include <string>                                 // for operator+, char_tr...
#include <vector>                                 // for vector
namespace htool {
template <typename CoefficientPrecision>
class DistributedOperator;
}

using namespace std;
using namespace htool;

template <typename solver_builder>
int test_solver_wo_overlap(int argc, char *argv[], int mu, char symmetric, char UPLO, std::string datapath) {

    // Get the number of processes
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Get the rank of the process
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    //
    bool test = 0;

    // HPDDM verbosity
    HPDDM::Option &opt = *HPDDM::Option::get();
    opt.parse(argc, argv, rank == 0);
    double tol = opt.val("tol", 1e-6);
    if (rank != 0)
        opt.remove("verbosity");
    opt.parse("-hpddm_max_it 200");

    // HTOOL
    double epsilon = tol;
    double eta     = 10;

    // Clustering
    if (rank == 0)
        std::cout << "Creating cluster tree" << std::endl;
    Cluster<double> target_cluster = read_cluster_tree<double>(datapath + "/cluster_" + NbrToStr(size) + "_cluster_tree_properties.csv", datapath + "/cluster_" + NbrToStr(size) + "_cluster_tree.csv");

    // Matrix
    if (rank == 0)
        std::cout << "Creating generators" << std::endl;
    Matrix<complex<double>> A;
    A.bytes_to_matrix(datapath + "/matrix.bin");
    GeneratorInUserNumberingFromMatrix<std::complex<double>> Generator(A);
    int n = A.nb_rows();

    // Right-hand side
    if (rank == 0)
        std::cout << "Building rhs" << std::endl;
    Matrix<complex<double>> f_global(n, mu);
    std::vector<complex<double>> temp(n);
    bytes_to_vector(temp, datapath + "/rhs.bin");
    for (int i = 0; i < mu; i++) {
        f_global.set_col(i, temp);
    }

    // Hmatrix
    if (rank == 0)
        std::cout << "Creating HMatrix" << std::endl;

    DefaultApproximationBuilder<Cplx, htool::underlying_type<Cplx>> default_build(Generator, target_cluster, target_cluster, epsilon, eta, symmetric, UPLO, MPI_COMM_WORLD);

    DistributedOperator<Cplx> &Operator        = default_build.distributed_operator;
    HMatrix<Cplx> local_block_diagonal_hmatrix = *default_build.block_diagonal_hmatrix;

    // Global vectors
    Matrix<complex<double>>
        x_global(n, mu), x_ref(n, mu), test_global(n, mu);
    bytes_to_vector(temp, datapath + "sol.bin");
    for (int i = 0; i < mu; i++) {
        x_ref.set_col(i, temp);
    }

    // Partition
    std::vector<int> cluster_to_ovr_subdomain;
    std::vector<int> ovr_subdomain_to_global;
    std::vector<int> neighbors;
    std::vector<std::vector<int>> intersections;
    bytes_to_vector(cluster_to_ovr_subdomain, datapath + "/cluster_to_ovr_subdomain_" + NbrToStr(size) + "_" + NbrToStr(rank) + ".bin");
    bytes_to_vector(ovr_subdomain_to_global, datapath + "/ovr_subdomain_to_global_" + NbrToStr(size) + "_" + NbrToStr(rank) + ".bin");
    bytes_to_vector(neighbors, datapath + "/neighbors_" + NbrToStr(size) + "_" + NbrToStr(rank) + ".bin");

    intersections.resize(neighbors.size());
    for (int p = 0; p < neighbors.size(); p++) {
        bytes_to_vector(intersections[p], datapath + "/intersections_" + NbrToStr(size) + "_" + NbrToStr(rank) + "_" + NbrToStr(p) + ".bin");
    }

    // Errors
    double error2;

    // Solve
    if (rank == 0)
        std::cout << "Creating Solver" << std::endl;
    solver_builder default_solver(Operator, local_block_diagonal_hmatrix);
    auto &block_jacobi_solver = default_solver.solver;

    print_hmatrix_information(local_block_diagonal_hmatrix, std::cout);
    // No precond wo overlap
    if (rank == 0)
        std::cout << "No precond without overlap:" << std::endl;

    opt.parse("-hpddm_schwarz_method none");
    block_jacobi_solver.solve(f_global.data(), x_global.data(), mu);
    block_jacobi_solver.print_infos();
    error2 = normFrob(f_global - A * x_global) / normFrob(f_global);
    if (rank == 0) {
        cout << "error: " << error2 << endl;
    }

    test = test || !(error2 < tol);

    x_global = 0;

    // DDM one level ASM wo overlap
    if (rank == 0)
        std::cout << "ASM one level without overlap:" << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);
    opt.parse("-hpddm_schwarz_method asm ");
    block_jacobi_solver.facto_one_level();
    block_jacobi_solver.solve(f_global.data(), x_global.data(), mu);
    block_jacobi_solver.print_infos();
    error2 = normFrob(f_global - A * x_global) / normFrob(f_global);

    if (rank == 0) {
        cout << "error: " << error2 << endl;
    }

    test = test || !(error2 < tol);

    x_global = 0;

    // DDM one level RAS wo overlap
    if (rank == 0)
        std::cout << "RAS one level without overlap:" << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);
    opt.parse("-hpddm_schwarz_method ras ");
    block_jacobi_solver.solve(f_global.data(), x_global.data(), mu);
    block_jacobi_solver.print_infos();
    error2 = normFrob(f_global - A * x_global) / normFrob(f_global);

    if (rank == 0) {
        cout << "error: " << error2 << endl;
    }

    test = test || !(error2 < tol);

    return test;
}
