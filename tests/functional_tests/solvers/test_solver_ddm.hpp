#include <htool/clustering/clustering.hpp>
#include <htool/distributed_operator/utility.hpp>
#include <htool/solvers/geneo/coarse_operator_builder.hpp>
#include <htool/solvers/geneo/coarse_space_builder.hpp>
#include <htool/solvers/utility.hpp>
#include <htool/testing/generator_test.hpp>
#include <htool/testing/gmsh.hpp>

using namespace std;
using namespace htool;

int test_solver_ddm(int argc, char *argv[], int mu, char data_symmetry, char symmetric, std::string datapath) {

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

    char UPLO = 'N';
    if (symmetric != 'N') {
        UPLO = 'L';
    }

    // Clustering
    if (rank == 0)
        std::cout << "Creating cluster tree" << std::endl;
    Cluster<double> target_cluster = read_cluster_tree<double>(datapath + "/cluster_" + NbrToStr(size) + "_cluster_tree_properties.csv", datapath + "/cluster_" + NbrToStr(size) + "_cluster_tree.csv");

    // Matrix
    if (rank == 0)
        std::cout << "Creating generators" << std::endl;
    Matrix<complex<double>> A;
    A.bytes_to_matrix(datapath + "/matrix.bin");
    GeneratorFromMatrix<std::complex<double>> Generator(A, target_cluster.get_permutation(), target_cluster.get_permutation());
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

    DefaultApproximationBuilder<std::complex<double>, htool::underlying_type<std::complex<double>>> default_build(Generator, target_cluster, target_cluster, epsilon, eta, symmetric, UPLO, MPI_COMM_WORLD);

    DistributedOperator<std::complex<double>> &Operator = default_build.distributed_operator;

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

    // Local numbering
    LocalNumberingBuilder local_numbering_builder(ovr_subdomain_to_global, cluster_to_ovr_subdomain, intersections);
    auto &renumbered_intersections  = local_numbering_builder.intersections;
    auto &local_to_global_numbering = local_numbering_builder.local_to_global_numbering;
    int local_size_wo_overlap       = Operator.get_target_partition().get_size_of_partition(rank);
    // int local_size_with_overlap = local_to_global_numbering.size();

    std::vector<double> geometry(n), local_geometry(local_to_global_numbering.size() * 3);
    bytes_to_vector(geometry, datapath + "/geometry.bin");
    for (int i = 0; i < local_to_global_numbering.size(); i++) {
        local_geometry[3 * i + 0] = geometry[3 * local_to_global_numbering[i] + 0];
        local_geometry[3 * i + 1] = geometry[3 * local_to_global_numbering[i] + 1];
        local_geometry[3 * i + 2] = geometry[3 * local_to_global_numbering[i] + 2];
    }

    ClusterTreeBuilder<double> recursive_build_strategy;
    // recursive_build_strategy.set_minclustersize(2);
    Cluster<double> local_cluster = recursive_build_strategy.create_cluster_tree(local_to_global_numbering.size(), 3, local_geometry.data(), 2, 2, nullptr);

    LocalGeneratorFromMatrix<std::complex<double>> local_generator(A, local_cluster.get_permutation(), local_cluster.get_permutation(), local_to_global_numbering, local_to_global_numbering);

    HMatrixTreeBuilder<std::complex<double>, double> local_hmatrix_builder(local_cluster, local_cluster, epsilon, eta, symmetric, UPLO, -1, -1, -1);

    HMatrix<std::complex<double>, double> local_hmatrix = local_hmatrix_builder.build(local_generator);
    print_distributed_hmatrix_information(local_hmatrix, std::cout, MPI_COMM_WORLD);
    save_leaves_with_rank(local_hmatrix, "local_hmatrix_" + std::to_string(rank));
    // Errors
    double error2;

    // Solve
    DefaultDDMSolverBuilder<std::complex<double>, double> default_ddm_solver(Operator, local_hmatrix, neighbors, renumbered_intersections);
    DDM<complex<double>> &ddm_with_overlap = default_ddm_solver.solver;

    // No precond with overlap
    if (rank == 0)
        std::cout << "No precond with overlap:" << std::endl;

    opt.parse("-hpddm_schwarz_method none");
    ddm_with_overlap.solve(f_global.data(), x_global.data(), mu);
    ddm_with_overlap.print_infos();
    error2 = normFrob(f_global - A * x_global) / normFrob(f_global);

    if (rank == 0) {
        cout << "error: " << error2 << endl;
    }

    test = test || !(error2 < tol);

    x_global = 0;

    // DDM one level ASM with overlap
    if (rank == 0)
        std::cout << "ASM one level with overlap:" << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);
    opt.parse("-hpddm_schwarz_method asm ");
    Matrix<complex<double>> Ki;

    if (data_symmetry == 'S' && size > 1) {
        opt.remove("geneo_threshold");
        opt.parse("-hpddm_geneo_nu 2");
        Ki.bytes_to_matrix(datapath + "/Ki_" + NbrToStr(size) + "_" + NbrToStr(rank) + ".bin");
        // ddm_with_overlap.build_coarse_space(Ki);

        auto geneo_coarse_space_dense_builder = GeneoCoarseSpaceDenseBuilder<std::complex<double>>::GeneoWithNu(local_size_wo_overlap, default_ddm_solver.block_diagonal_dense_matrix, Ki, symmetric, UPLO, 2);
        // geneo_coarse_space_dense_builder.set_geneo_nu(2);
        GeneoCoarseOperatorBuilder<std::complex<double>> geneo_coarse_operator_builder(Operator);
        ddm_with_overlap.build_coarse_space(geneo_coarse_space_dense_builder, geneo_coarse_operator_builder);
    }

    ddm_with_overlap.facto_one_level();
    ddm_with_overlap.solve(f_global.data(), x_global.data(), mu);
    ddm_with_overlap.print_infos();
    error2 = normFrob(f_global - A * x_global) / normFrob(f_global);

    if (rank == 0) {
        cout << "error: " << error2 << endl;
    }

    test = test || !(error2 < tol);

    x_global = 0;

    // DDM one level RAS with overlap
    if (rank == 0)
        std::cout << "RAS one level with overlap:" << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);
    opt.parse("-hpddm_schwarz_method ras ");
    ddm_with_overlap.solve(f_global.data(), x_global.data(), mu);
    ddm_with_overlap.print_infos();
    error2 = normFrob(f_global - A * x_global) / normFrob(f_global);
    if (rank == 0) {
        cout << "error: " << error2 << endl;
    }

    test = test || !(error2 < tol);

    x_global = 0;

    // DDM two level ASM with overlap
    if (data_symmetry == 'S' && size > 1) {
        if (rank == 0)
            std::cout << "ASM two level with overlap:" << std::endl;
        MPI_Barrier(MPI_COMM_WORLD);
        opt.parse("-hpddm_schwarz_method asm -hpddm_schwarz_coarse_correction additive");
        ddm_with_overlap.solve(f_global.data(), x_global.data(), mu);
        ddm_with_overlap.print_infos();
        error2 = normFrob(f_global - A * x_global) / normFrob(f_global);
        if (rank == 0) {
            cout << "error: " << error2 << endl;
        }

        test = test || !(error2 < tol);

        x_global = 0;

        if (rank == 0)
            std::cout << "RAS two level with overlap:" << std::endl;
        MPI_Barrier(MPI_COMM_WORLD);
        opt.parse("-hpddm_schwarz_method ras -hpddm_schwarz_coarse_correction additive");
        ddm_with_overlap.solve(f_global.data(), x_global.data(), mu);
        ddm_with_overlap.print_infos();
        ddm_with_overlap.clean();
        error2 = normFrob(f_global - A * x_global) / normFrob(f_global);
        if (rank == 0) {
            cout << "error: " << error2 << endl;
        }

        test = test || !(error2 < tol);

        x_global = 0;

        // DDM solver with threshold
        if (rank == 0)
            std::cout << "RAS two level with overlap and threshold:" << std::endl;
        opt.parse("-hpddm_schwarz_method ras -hpddm_schwarz_coarse_correction additive");
        // DDM<complex<double>> ddm_with_overlap_threshold(Generator, &Operator, ovr_subdomain_to_global, cluster_to_ovr_subdomain, neighbors, intersections);
        DefaultDDMSolverBuilder<complex<double>, double> default_ddm_solver_with_threshold(Operator, local_hmatrix, neighbors, renumbered_intersections);
        DDM<complex<double>> &ddm_with_overlap_threshold = default_ddm_solver_with_threshold.solver; // build_ddm_solver(Operator, local_block_diagonal_hmatrix, Generator, ovr_subdomain_to_global, cluster_to_ovr_subdomain, neighbors, intersections);
        Ki.bytes_to_matrix(datapath + "/Ki_" + NbrToStr(size) + "_" + NbrToStr(rank) + ".bin");
        // ddm_with_overlap_threshold.build_coarse_space(Ki);
        auto geneo_coarse_space_dense_builder = GeneoCoarseSpaceDenseBuilder<std::complex<double>>::GeneoWithThreshold(local_size_wo_overlap, default_ddm_solver.block_diagonal_dense_matrix, Ki, symmetric, UPLO, 100);
        // geneo_coarse_space_dense_builder.set_geneo_threshold(100);
        GeneoCoarseOperatorBuilder<std::complex<double>> geneo_coarse_operator_builder(Operator);
        ddm_with_overlap_threshold.build_coarse_space(geneo_coarse_space_dense_builder, geneo_coarse_operator_builder);
        ddm_with_overlap_threshold.facto_one_level();
        ddm_with_overlap_threshold.solve(f_global.data(), x_global.data(), mu);
        ddm_with_overlap_threshold.print_infos();
        ddm_with_overlap_threshold.clean();
        error2 = normFrob(f_global - A * x_global) / normFrob(f_global);
        if (rank == 0) {
            cout << "error: " << error2 << endl;
        }

        test = test || !(error2 < tol);

        x_global = 0;
    }

    return test;
}
