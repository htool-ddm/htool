#include <htool/clustering/clustering.hpp>
#include <htool/distributed_operator/utility.hpp>
#include <htool/solvers/utility.hpp>
#include <htool/testing/generator_test.hpp>
#include <htool/testing/gmsh.hpp>

using namespace std;
using namespace htool;

int test_solver_ddm(int argc, char *argv[], int mu, char symmetric, bool) {

    // Input file
    if (argc < 2) { // argc should be 5 or more for correct execution
        // We print argv[0] assuming it is the program name
        cout << "usage: " << argv[0] << " datapath\n"; // LCOV_EXCL_LINE
        return 1;                                      // LCOV_EXCL_LINE
    }
    string datapath = argv[1];

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

    // std::vector<int>tab(n);
    // std::iota(tab.begin(),tab.end(),int(0));
    // t->build(p,std::vector<double>(n,0),tab,std::vector<double>(n,1),2);
    // std::vector<int> permutation_test(n);
    // bytes_to_vector(permutation_test,datapath+"permutation.bin");
    // t->save(p,"test_cluster",{0,1,2,3});

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

    const HMatrix<Cplx> *local_block_diagonal_hmatrix = nullptr;

    DefaultApproximationBuilder<Cplx, htool::underlying_type<Cplx>> default_build(Generator, target_cluster, target_cluster, epsilon, eta, symmetric, UPLO, MPI_COMM_WORLD);

    DistributedOperator<Cplx> &Operator = default_build.distributed_operator;
    local_block_diagonal_hmatrix        = default_build.block_diagonal_hmatrix;

    // if (off_diagonal_approximation) {
    //     // Setup data for off diagonal geometry
    //     int off_diagonal_nr, off_diagonal_nc, nc_left, nc_local, nc_right;
    //     HA.get_off_diagonal_size(off_diagonal_nr, off_diagonal_nc);

    //     vector<double> off_diagonal_p1(off_diagonal_nr * t->get_space_dim());
    //     vector<double> off_diagonal_p2(off_diagonal_nc * t->get_space_dim());

    //     HA.get_off_diagonal_geometries(p.data()->data(), p.data()->data(), off_diagonal_p1.data(), off_diagonal_p2.data());

    //     // Clustering
    //     std::shared_ptr<VirtualCluster> new_cluster_target = std::make_shared<Cluster<PCAGeometricClustering>>();
    //     std::shared_ptr<VirtualCluster> new_cluster_source = std::make_shared<Cluster<PCAGeometricClustering>>();
    //     new_cluster_target->build(off_diagonal_nr, off_diagonal_p1.data(), 2, MPI_COMM_SELF);
    //     new_cluster_source->build(off_diagonal_nc, off_diagonal_p2.data(), 2, MPI_COMM_SELF);

    //     // Generator
    //     Matrix<complex<double>> off_diagonal_A(off_diagonal_nr, off_diagonal_nc);
    //     for (int i = t->get_local_offset(); i < t->get_local_offset() + t->get_local_size(); i++) {
    //         for (int j = 0; j < t->get_local_offset(); j++) {
    //             off_diagonal_A(i - t->get_local_offset(), j) = A(t->get_global_perm(i), t->get_global_perm(j));
    //         }
    //     }
    //     for (int i = t->get_local_offset(); i < t->get_local_offset() + t->get_local_size(); i++) {
    //         for (int j = t->get_local_offset() + t->get_local_size(); j < t->get_size(); j++) {
    //             off_diagonal_A(i - t->get_local_offset(), j - t->get_local_size()) = A(t->get_global_perm(i), t->get_global_perm(j));
    //         }
    //     }

    //     GeneratorFromMatrix<complex<double>>
    //         off_diagonal_generator(off_diagonal_A);

    //     // OffDiagonalHmatrix
    //     VirtualHMatrix<complex<double>> *HA_ptr = &HA;
    //     auto OffDiagonalHA                      = std::make_shared<OffDiagonalApproximationWithHMatrix<complex<double>>>(HA_ptr, new_cluster_target, new_cluster_source);
    //     OffDiagonalHA->set_compression(compressor);
    //     OffDiagonalHA->build(off_diagonal_generator, off_diagonal_p1.data(), off_diagonal_p2.data());

    //     HA.set_off_diagonal_approximation(std::shared_ptr<VirtualOffDiagonalApproximation<complex<double>>>(OffDiagonalHA));
    // }

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
    bytes_to_vector(cluster_to_ovr_subdomain, datapath + "cluster_to_ovr_subdomain_" + NbrToStr(size) + "_" + NbrToStr(rank) + ".bin");
    bytes_to_vector(ovr_subdomain_to_global, datapath + "ovr_subdomain_to_global_" + NbrToStr(size) + "_" + NbrToStr(rank) + ".bin");
    bytes_to_vector(neighbors, datapath + "neighbors_" + NbrToStr(size) + "_" + NbrToStr(rank) + ".bin");

    intersections.resize(neighbors.size());
    for (int p = 0; p < neighbors.size(); p++) {
        bytes_to_vector(intersections[p], datapath + "intersections_" + NbrToStr(size) + "_" + NbrToStr(rank) + "_" + NbrToStr(p) + ".bin");
    }

    // Errors
    double error2;

    // Solve
    DefaultSolverBuilder<complex<double>, double> default_solver(Operator, local_block_diagonal_hmatrix);
    DefaultDDMSolverBuilder<complex<double>, double> default_ddm_solver(Operator, local_block_diagonal_hmatrix, Generator, ovr_subdomain_to_global, cluster_to_ovr_subdomain, neighbors, intersections);
    DDM<complex<double>> &ddm_wo_overlap   = default_solver.solver;     //  = build_ddm_solver(Operator, local_block_diagonal_hmatrix);
    DDM<complex<double>> &ddm_with_overlap = default_ddm_solver.solver; // build_ddm_solver(Operator, local_block_diagonal_hmatrix, Generator, ovr_subdomain_to_global, cluster_to_ovr_subdomain, neighbors, intersections);

    // Test renum_to_global
    std::vector<int> renum(ovr_subdomain_to_global.size(), -1);
    std::vector<int> renum_to_global(ovr_subdomain_to_global.size());
    for (int i = 0; i < cluster_to_ovr_subdomain.size(); i++) {
        renum[cluster_to_ovr_subdomain[i]] = i;
        renum_to_global[i]                 = ovr_subdomain_to_global[cluster_to_ovr_subdomain[i]];
    }
    int count = cluster_to_ovr_subdomain.size();
    for (int i = 0; i < ovr_subdomain_to_global.size(); i++) {
        if (renum[i] == -1) {
            renum[i]                 = count;
            renum_to_global[count++] = ovr_subdomain_to_global[i];
        }
    }

    // std::vector<int> renum_to_global_b = ddm_with_overlap.get_local_to_global_numbering();
    // test                               = test || !(norm2(renum_to_global_b - renum_to_global) < 1e-10);
    // test                               = test || !(ddm_with_overlap.get_local_size() == ovr_subdomain_to_global.size());

    // No precond wo overlap
    if (rank == 0)
        std::cout << "No precond without overlap:" << std::endl;

    opt.parse("-hpddm_schwarz_method none");
    ddm_wo_overlap.solve(f_global.data(), x_global.data(), mu);
    ddm_wo_overlap.print_infos();
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
    ddm_wo_overlap.facto_one_level();
    ddm_wo_overlap.solve(f_global.data(), x_global.data(), mu);
    ddm_wo_overlap.print_infos();
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
    ddm_wo_overlap.solve(f_global.data(), x_global.data(), mu);
    ddm_wo_overlap.print_infos();
    error2 = normFrob(f_global - A * x_global) / normFrob(f_global);

    if (rank == 0) {
        cout << "error: " << error2 << endl;
    }

    test = test || !(error2 < tol);

    x_global = 0;

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
    if (symmetric == 'S' && size > 1) {
        opt.remove("geneo_threshold");
        opt.parse("-hpddm_geneo_nu 2");
        Ki.bytes_to_matrix(datapath + "/Ki_" + NbrToStr(size) + "_" + NbrToStr(rank) + ".bin");
        ddm_with_overlap.build_coarse_space(Ki);
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
    if (symmetric == 'S' && size > 1) {
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
        opt.parse("-hpddm_schwarz_method ras -hpddm_schwarz_coarse_correction additive -hpddm_geneo_threshold 100");
        // DDM<complex<double>> ddm_with_overlap_threshold(Generator, &Operator, ovr_subdomain_to_global, cluster_to_ovr_subdomain, neighbors, intersections);
        DefaultDDMSolverBuilder<complex<double>, double> default_ddm_solver_with_threshold(Operator, local_block_diagonal_hmatrix, Generator, ovr_subdomain_to_global, cluster_to_ovr_subdomain, neighbors, intersections);
        DDM<complex<double>> &ddm_with_overlap_threshold = default_ddm_solver_with_threshold.solver; // build_ddm_solver(Operator, local_block_diagonal_hmatrix, Generator, ovr_subdomain_to_global, cluster_to_ovr_subdomain, neighbors, intersections);
        Ki.bytes_to_matrix(datapath + "/Ki_" + NbrToStr(size) + "_" + NbrToStr(rank) + ".bin");
        ddm_with_overlap_threshold.build_coarse_space(Ki);
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
