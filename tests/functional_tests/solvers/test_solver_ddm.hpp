#include <algorithm>                                           // for max
#include <complex>                                             // for complex
#include <htool/basic_types/vector.hpp>                        // for bytes...
#include <htool/clustering/cluster_node.hpp>                   // for Cluster
#include <htool/clustering/cluster_output.hpp>                 // for read_...
#include <htool/distributed_operator/distributed_operator.hpp> // for Distr...
#include <htool/distributed_operator/utility.hpp>              // for Defau...
#include <htool/matrix/matrix.hpp>                             // for Matrix
#include <htool/misc/misc.hpp>                                 // for under...
#include <htool/misc/user.hpp>                                 // for NbrToStr
#include <htool/solvers/ddm.hpp>                               // for DDM
#include <htool/solvers/geneo/coarse_operator_builder.hpp>     // for Geneo...
#include <htool/solvers/geneo/coarse_space_builder.hpp>        // for Geneo...
#include <htool/solvers/utility.hpp>                           // for DDMSo...
#include <htool/testing/generator_test.hpp>                    // for Gener...
#include <initializer_list>                                    // for initi...
#include <iostream>                                            // for basic...
#include <memory>                                              // for alloc...
#include <mpi.h>                                               // for MPI_B...
#include <string>                                              // for opera...
#include <vector>                                              // for vector

using namespace std;
using namespace htool;

template <typename CoefficientPrecision, typename CoordinatePrecision, typename solver_builder>
int test_solver_ddm(int argc, char *argv[], int mu, char data_symmetry, char symmetric, char UPLO, std::string datapath) {

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
    CoordinatePrecision tol = opt.val("tol", 1e-6);
    if (rank != 0)
        opt.remove("verbosity");
    opt.parse("-hpddm_max_it 200");

    // HTOOL
    CoordinatePrecision epsilon = tol;
    // epsilon        = 1e-10;
    CoordinatePrecision eta = 10;

    // Clustering
    if (rank == 0)
        std::cout << "Creating cluster tree" << std::endl;
    Cluster<CoordinatePrecision> target_cluster = read_cluster_tree<CoordinatePrecision>(datapath + "/cluster_" + NbrToStr(size) + "_cluster_tree_properties.csv", datapath + "/cluster_" + NbrToStr(size) + "_cluster_tree.csv");

    // Matrix
    if (rank == 0)
        std::cout << "Creating generators" << std::endl;
    std::unique_ptr<VirtualGenerator<CoefficientPrecision>> generator;
    Matrix<CoefficientPrecision> A;
    Matrix<std::complex<double>> A_original;
    A_original.bytes_to_matrix(datapath + "/matrix.bin");
    if constexpr (htool::is_complex<CoefficientPrecision>()) {
        generator = std::make_unique<GeneratorInUserNumberingFromMatrix<std::complex<double>>>(A_original);
        A         = A_original;
    } else {
        generator = std::make_unique<GeneratorInUserNumberingFromMatrixToReal<double>>(A_original);
        A.resize(A_original.nb_rows(), A_original.nb_cols());
        for (int i = 0; i < A_original.nb_rows(); i++) {
            for (int j = 0; j < A_original.nb_cols(); j++) {
                A(i, j) = A_original(i, j).real();
            }
        }
    }
    int n = A.nb_rows();

    // Right-hand side
    if (rank == 0)
        std::cout << "Building rhs" << std::endl;
    Matrix<CoefficientPrecision> f_global(n, mu);
    std::vector<complex<double>> temp(n);
    bytes_to_vector(temp, datapath + "/rhs.bin");
    for (int i = 0; i < mu; i++) {
        if constexpr (htool::is_complex<CoefficientPrecision>()) {
            f_global.set_col(i, temp);
        } else {
            for (int j = 0; j < f_global.nb_rows(); j++) {
                f_global(j, i) = temp[j].real();
            }
        }
    }

    // Hmatrix
    if (rank == 0)
        std::cout << "Creating HMatrix" << std::endl;

    DefaultApproximationBuilder<CoefficientPrecision, htool::underlying_type<CoefficientPrecision>> default_build(*generator, target_cluster, target_cluster, HMatrixTreeBuilder<CoefficientPrecision>(epsilon, eta, symmetric, UPLO), MPI_COMM_WORLD);

    DistributedOperator<CoefficientPrecision> &Operator = default_build.distributed_operator;

    // Global vectors
    Matrix<CoefficientPrecision> x_global(n, mu), x_ref(n, mu), test_global(n, mu);
    bytes_to_vector(temp, datapath + "sol.bin");
    for (int i = 0; i < mu; i++) {
        if constexpr (htool::is_complex<CoefficientPrecision>()) {
            x_ref.set_col(i, temp);
        } else {
            for (int j = 0; j < x_ref.nb_rows(); j++) {
                x_ref(j, i) = temp[j].real();
            }
        }
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

    // Error
    double error2;

    // Solve
    if (rank == 0)
        std::cout << "Creating HMatrix" << std::endl;
    std::vector<CoordinatePrecision> geometry(n);
    bytes_to_vector(geometry, datapath + "/geometry.bin");
    std::unique_ptr<solver_builder> default_ddm_solver_ptr;
    if constexpr (std::is_same_v<solver_builder, DDMSolverWithDenseLocalSolver<CoefficientPrecision, CoordinatePrecision>>) {
        default_ddm_solver_ptr = std::make_unique<solver_builder>(Operator, ovr_subdomain_to_global, cluster_to_ovr_subdomain, neighbors, intersections, *generator, 3, geometry.data(), epsilon, eta);
    } else {
        default_ddm_solver_ptr = std::make_unique<solver_builder>(Operator, ovr_subdomain_to_global, cluster_to_ovr_subdomain, neighbors, intersections, *generator, 3, geometry.data(), ClusterTreeBuilder<CoordinatePrecision>(), HMatrixTreeBuilder<CoefficientPrecision>(epsilon, eta, symmetric, UPLO));
    }
    auto &default_ddm_solver = *default_ddm_solver_ptr;
    auto &ddm_with_overlap   = default_ddm_solver.solver;

    // DDMSolverBuilder<complex<double>> default_ddm_solver(Operator, ovr_subdomain_to_global, cluster_to_ovr_subdomain, neighbors, intersections, Generator, 3, geometry.data(), ClusterTreeBuilder<double>(), HMatrixTreeBuilder<complex<double>>(epsilon, eta, symmetric, UPLO));

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
    Matrix<CoefficientPrecision> Ki;
    if constexpr (!htool::is_complex<CoefficientPrecision>()) {
        if (data_symmetry == 'S' && size > 1) {
            opt.remove("geneo_threshold");
            opt.parse("-hpddm_geneo_nu 2");
            Matrix<std::complex<double>> tmp;
            tmp.bytes_to_matrix(datapath + "/Ki_" + NbrToStr(size) + "_" + NbrToStr(rank) + ".bin");
            if constexpr (htool::is_complex<CoefficientPrecision>()) {
                Ki = tmp;
            } else {
                Ki.resize(tmp.nb_rows(), tmp.nb_cols());
                for (int i = 0; i < tmp.nb_rows(); i++) {
                    for (int j = 0; j < tmp.nb_cols(); j++) {
                        Ki(i, j) = tmp(i, j).real();
                    }
                }
            }
            int local_size_wo_overlap   = Operator.get_target_partition().get_size_of_partition(rank);
            int local_size_with_overlap = Ki.nb_cols();
            std::unique_ptr<GeneoCoarseSpaceDenseBuilder<CoefficientPrecision>> geneo_coarse_space_dense_builder;
            if constexpr (std::is_same_v<solver_builder, DDMSolverWithDenseLocalSolver<CoefficientPrecision, CoordinatePrecision>>) {
                geneo_coarse_space_dense_builder = std::make_unique<GeneoCoarseSpaceDenseBuilder<CoefficientPrecision>>(GeneoCoarseSpaceDenseBuilder<CoefficientPrecision>::GeneoWithNu(local_size_wo_overlap, local_size_with_overlap, default_ddm_solver.block_diagonal_dense_matrix, Ki, symmetric, UPLO, 2));
            } else {
                geneo_coarse_space_dense_builder = std::make_unique<GeneoCoarseSpaceDenseBuilder<CoefficientPrecision>>(GeneoCoarseSpaceDenseBuilder<CoefficientPrecision>::GeneoWithNu(local_size_wo_overlap, local_size_with_overlap, *default_build.block_diagonal_hmatrix, Ki, symmetric, UPLO, 2));
            }
            GeneoCoarseOperatorBuilder<CoefficientPrecision> geneo_coarse_operator_builder(Operator);
            ddm_with_overlap.build_coarse_space(*geneo_coarse_space_dense_builder, geneo_coarse_operator_builder);
        }
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
    if constexpr (!htool::is_complex<CoefficientPrecision>()) {
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

            std::unique_ptr<solver_builder> default_ddm_solver_with_threshold_ptr;
            if constexpr (std::is_same_v<solver_builder, DDMSolverWithDenseLocalSolver<CoefficientPrecision, CoordinatePrecision>>) {
                default_ddm_solver_with_threshold_ptr = std::make_unique<solver_builder>(Operator, ovr_subdomain_to_global, cluster_to_ovr_subdomain, neighbors, intersections, *generator, 3, geometry.data(), epsilon, eta);
            } else {
                default_ddm_solver_with_threshold_ptr = std::make_unique<solver_builder>(Operator, ovr_subdomain_to_global, cluster_to_ovr_subdomain, neighbors, intersections, *generator, 3, geometry.data(), ClusterTreeBuilder<CoordinatePrecision>(), HMatrixTreeBuilder<CoefficientPrecision>(epsilon, eta, symmetric, UPLO));
            }
            auto &default_ddm_solver_with_threshold = *default_ddm_solver_with_threshold_ptr;

            auto &ddm_with_overlap_threshold = default_ddm_solver_with_threshold.solver;
            Matrix<std::complex<double>> tmp;
            tmp.bytes_to_matrix(datapath + "/Ki_" + NbrToStr(size) + "_" + NbrToStr(rank) + ".bin");
            if constexpr (htool::is_complex<CoefficientPrecision>()) {
                Ki = tmp;
            } else {
                Ki.resize(tmp.nb_rows(), tmp.nb_cols());
                for (int i = 0; i < tmp.nb_rows(); i++) {
                    for (int j = 0; j < tmp.nb_cols(); j++) {
                        Ki(i, j) = tmp(i, j).real();
                    }
                }
            }
            int local_size_wo_overlap   = Operator.get_target_partition().get_size_of_partition(rank);
            int local_size_with_overlap = Ki.nb_cols();
            std::unique_ptr<GeneoCoarseSpaceDenseBuilder<CoefficientPrecision>> geneo_coarse_space_dense_builder;
            if constexpr (std::is_same_v<solver_builder, DDMSolverWithDenseLocalSolver<CoefficientPrecision, CoordinatePrecision>>) {
                geneo_coarse_space_dense_builder = std::make_unique<GeneoCoarseSpaceDenseBuilder<CoefficientPrecision>>(GeneoCoarseSpaceDenseBuilder<CoefficientPrecision>::GeneoWithThreshold(local_size_wo_overlap, local_size_with_overlap, default_ddm_solver.block_diagonal_dense_matrix, Ki, symmetric, UPLO, 100));
            } else {
                geneo_coarse_space_dense_builder = std::make_unique<GeneoCoarseSpaceDenseBuilder<CoefficientPrecision>>(GeneoCoarseSpaceDenseBuilder<CoefficientPrecision>::GeneoWithThreshold(local_size_wo_overlap, local_size_with_overlap, *default_build.block_diagonal_hmatrix, Ki, symmetric, UPLO, 100));
            }
            // geneo_coarse_space_dense_builder.set_geneo_threshold(100);
            GeneoCoarseOperatorBuilder<CoefficientPrecision> geneo_coarse_operator_builder(Operator);
            ddm_with_overlap_threshold.build_coarse_space(*geneo_coarse_space_dense_builder, geneo_coarse_operator_builder);
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
    }

    return test;
}
