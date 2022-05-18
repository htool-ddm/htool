#include <htool/clustering/pca.hpp>
#include <htool/lrmat/fullACA.hpp>
#include <htool/testing/generator_input.hpp>
#include <htool/testing/generator_test.hpp>
#include <htool/testing/geometry.hpp>
#include <htool/types/hmatrix.hpp>
#include <htool/types/off_diagonal_approximation_with_hmatrix.hpp>

using namespace std;
using namespace htool;

template <typename T, typename GeneratorTestType, typename ClusterImpl, template <typename> class CompressionImpl, template <typename> class HMatrixImpl>
bool test_virtual_hmat_product(int nr, int nc, int mu, bool use_permutation, char Symmetry, char UPLO, char op, bool off_diagonal_approximation) {

    // // Initialize the MPI environment
    // MPI_Init(&argc, &argv);

    // Get the number of processes
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Get the rank of the process
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    //
    bool test = 0;
    int ndistance;
    std::vector<double> distance;

    if (Symmetry == 'N') {
        ndistance = 4;
        distance.resize(ndistance);
        distance[0] = 3;
        distance[1] = 5;
        distance[2] = 7;
        distance[3] = 10;
    } else {
        ndistance = 1;
        distance.resize(ndistance);
    }

    double epsilon = 1e-3;
    double eta     = 10;

    for (int idist = 0; idist < distance.size(); idist++) {

        srand(1);
        // we set a constant seed for rand because we want always the same result if we run the check many times
        // (two different initializations with the same seed will generate the same succession of results in the subsequent calls to rand)

        double z1 = 1;
        vector<double> p1(3 * nr), p1_perm, off_diagonal_p1;
        vector<double> p2(Symmetry == 'N' ? 3 * nc : 1), p2_perm, off_diagonal_p2;
        create_disk(3, z1, nr, p1.data());
        int size_numbering = nr / (size);
        int count_size     = 0;
        std::vector<int> MasterOffset_target, MasterOffset_source;
        for (int p = 0; p < size - 1; p++) {
            MasterOffset_target.push_back(count_size);
            MasterOffset_target.push_back(size_numbering);

            count_size += size_numbering;
        }
        MasterOffset_target.push_back(count_size);
        MasterOffset_target.push_back(nr - count_size);

        size_numbering = nc / size;
        count_size     = 0;

        std::shared_ptr<VirtualCluster> s;
        std::shared_ptr<VirtualCluster> t = make_shared<ClusterImpl>();
        t->build(nr, p1.data(), MasterOffset_target.data(), 2);

        if (Symmetry == 'N') {
            double z2 = 1 + distance[idist];
            create_disk(3, z2, nc, p2.data());

            for (int p = 0; p < size - 1; p++) {
                MasterOffset_source.push_back(count_size);
                MasterOffset_source.push_back(size_numbering);

                count_size += size_numbering;
            }
            MasterOffset_source.push_back(count_size);
            MasterOffset_source.push_back(nc - count_size);
            s = make_shared<ClusterImpl>();
            s->build(nc, p2.data(), MasterOffset_source.data(), 2);
        } else {

            MasterOffset_source = MasterOffset_target;

            s  = t;
            p2 = p1;
        }

        // Permutation on geometry
        if (!use_permutation) {
            p1_perm.resize(3 * nr);
            for (int i = 0; i < t->get_global_perm().size(); i++) {
                p1_perm[i * 3 + 0] = p1[t->get_global_perm(i) * 3 + 0];
                p1_perm[i * 3 + 1] = p1[t->get_global_perm(i) * 3 + 1];
                p1_perm[i * 3 + 2] = p1[t->get_global_perm(i) * 3 + 2];
            }
            p2_perm.resize(3 * nc);
            if (Symmetry == 'N') {
                for (int i = 0; i < s->get_global_perm().size(); i++) {
                    p2_perm[i * 3 + 0] = p2[s->get_global_perm(i) * 3 + 0];
                    p2_perm[i * 3 + 1] = p2[s->get_global_perm(i) * 3 + 1];
                    p2_perm[i * 3 + 2] = p2[s->get_global_perm(i) * 3 + 2];
                }
            } else {
                p2_perm = p1_perm;
            }
        }

        // Generator
        GeneratorTestType A(3, nr, nc, p1, p2);
        GeneratorTestType A_perm(3, nr, nc, p1_perm, p2_perm);

        // Hmatrix
        std::shared_ptr<VirtualLowRankGenerator<T>> compressor = std::make_shared<CompressionImpl<T>>();

        std::shared_ptr<VirtualHMatrix<T>> HA = std::make_shared<HMatrixImpl<T>>(t, s, epsilon, eta, Symmetry, UPLO);
        HA->set_compression(compressor);
        HA->set_use_permutation(use_permutation);

        if (off_diagonal_approximation) {

            // Setup data for off diagonal geometry
            int off_diagonal_nr, off_diagonal_nc, nc_left, nc_local, nc_right;
            HA->get_off_diagonal_size(off_diagonal_nr, off_diagonal_nc);

            vector<double> off_diagonal_p1(off_diagonal_nr * t->get_space_dim());
            vector<double> off_diagonal_p2(off_diagonal_nc * s->get_space_dim());
            if (use_permutation) {
                HA->get_off_diagonal_geometries(p1.data(), p2.data(), off_diagonal_p1.data(), off_diagonal_p2.data());
            } else {
                HA->get_off_diagonal_geometries(p1_perm.data(), p2_perm.data(), off_diagonal_p1.data(), off_diagonal_p2.data());
            }

            // Clustering
            std::shared_ptr<VirtualCluster> new_cluster_target = std::make_shared<ClusterImpl>();
            std::shared_ptr<VirtualCluster> new_cluster_source = std::make_shared<ClusterImpl>();
            new_cluster_target->build(off_diagonal_nr, off_diagonal_p1.data(), 2, MPI_COMM_SELF);
            new_cluster_source->build(off_diagonal_nc, off_diagonal_p2.data(), 2, MPI_COMM_SELF);

            // Generator
            GeneratorTestType off_diagonal_generator(3, off_diagonal_nr, off_diagonal_nc, off_diagonal_p1, off_diagonal_p2);

            // OffDiagonalHmatrix
            auto OffDiagonalHA = std::make_shared<OffDiagonalApproximationWithHMatrix<T>>(HA.get(), new_cluster_target, new_cluster_source);
            OffDiagonalHA->set_compression(compressor);
            OffDiagonalHA->build(off_diagonal_generator, off_diagonal_p1.data(), off_diagonal_p2.data());
            if (rank == 0)
                OffDiagonalHA->get_HMatrix()->print_infos();
            HA->set_off_diagonal_approximation(std::shared_ptr<VirtualOffDiagonalApproximation<T>>(OffDiagonalHA));
        }

        double time = MPI_Wtime();
        if (use_permutation) {
            if (Symmetry == 'N')
                HA->build(A, p1.data(), p2.data());
            else
                HA->build(A, p1.data());
        } else {
            if (Symmetry == 'N')
                HA->build(A_perm, p1_perm.data(), p2_perm.data());
            else
                HA->build(A_perm, p1_perm.data());
        }

        std::cout << MPI_Wtime() - time << std::endl;
        time = MPI_Wtime() - time;
        HA->print_infos();

        // Input sizes
        int ni = (op == 'T' || op == 'C') ? nr : nc;
        int no = (op == 'T' || op == 'C') ? nc : nr;

        std::vector<int> MasterOffset_input  = (op == 'T' || op == 'C') ? MasterOffset_target : MasterOffset_source;
        std::vector<int> MasterOffset_output = (op == 'T' || op == 'C') ? MasterOffset_source : MasterOffset_target;

        // Random vector
        vector<T> random_vector(ni * mu, 1);
        if (rank == 0) {
            generate_random_vector(random_vector);
        }
        MPI_Bcast(random_vector.data(), random_vector.size(), wrapper_mpi<T>::mpi_type(), 0, MPI_COMM_WORLD);

        std::vector<T> x_global(ni * mu, 1);
        if (use_permutation) {
            x_global = random_vector;
        } else {
            for (int j = 0; j < mu; j++) {
                if (op == 'T' || op == 'C') {
                    global_to_cluster(t.get(), random_vector.data() + ni * j, x_global.data() + ni * j);
                } else {
                    global_to_cluster(s.get(), random_vector.data() + ni * j, x_global.data() + ni * j);
                }
            }
        }

        // Global vectors
        std::vector<T> f_global(no * mu), buffer, f_global_test(no * mu);
        if (op == 'T') {
            A.mvprod_transp(random_vector.data(), f_global.data(), mu);
        } else if (op == 'C') {
            // A.mvprod_conj(random_vector.data(), f_global.data(), mu);
        } else {
            A.mvprod(random_vector.data(), f_global.data(), mu);
        }

        std::cout << MPI_Wtime() - time << std::endl;
        time = MPI_Wtime() - time;

        // Global product
        if (use_permutation) {
            if (op == 'T') {
                HA->mvprod_transp_global_to_global(x_global.data(), f_global_test.data(), mu);
            } else if (op == 'C') {
                // HA->mvprod_conj_global_to_global(x_global.data(), f_global_test.data(), mu);
            } else {
                HA->mvprod_global_to_global(x_global.data(), f_global_test.data(), mu);
            }
        } else {
            buffer.resize(no * mu);
            if (op == 'T') {
                HA->mvprod_transp_global_to_global(x_global.data(), buffer.data(), mu);
            } else if (op == 'C') {
                // HA->mvprod_conj_global_to_global(x_global.data(), buffer.data(), mu);
            } else {
                HA->mvprod_global_to_global(x_global.data(), buffer.data(), mu);
            }
            for (int j = 0; j < mu; j++) {
                if (op == 'T' || op == 'C') {
                    cluster_to_global(s.get(), buffer.data() + no * j, f_global_test.data() + no * j);
                } else {
                    cluster_to_global(t.get(), buffer.data() + no * j, f_global_test.data() + no * j);
                }
            }
        }

        std::cout << MPI_Wtime() - time << std::endl;
        time = MPI_Wtime() - time;

        // Errors
        double global_diff = norm2(f_global - f_global_test) / norm2(f_global);

        if (rank == 0) {
            cout << "difference on hmat product computed globally: " << global_diff << endl;
        }
        test = test || !(global_diff < HA->get_epsilon());

        // Local vectors
        std::vector<T> x_local(MasterOffset_input[2 * rank + 1] * mu), f_local(MasterOffset_output[2 * rank + 1] * mu), f_local_to_global(no * mu);
        for (int i = 0; i < mu; i++) {
            std::copy_n(x_global.data() + MasterOffset_input[2 * rank] + ni * i, MasterOffset_input[2 * rank + 1], x_local.data() + MasterOffset_input[2 * rank + 1] * i);
        }

        // Local product
        if (use_permutation) {
            if (op == 'T') {
                HA->mvprod_transp_local_to_local(x_local.data(), f_local.data(), mu);
            } else if (op == 'C') {
                // HA->mvprod_conj_local_to_local(x_local.data(), f_local.data(), mu);
            } else {
                HA->mvprod_local_to_local(x_local.data(), f_local.data(), mu);
            }

        } else {
            buffer.resize(MasterOffset_output[2 * rank + 1] * mu);
            if (op == 'T') {
                HA->mvprod_transp_local_to_local(x_local.data(), buffer.data(), mu);
            } else if (op == 'C') {
                // HA->mvprod_conj_local_to_local(x_local.data(), buffer.data(), mu);
            } else {
                HA->mvprod_local_to_local(x_local.data(), buffer.data(), mu);
            }
            for (int j = 0; j < mu; j++) {
                if (op == 'T' || op == 'C') {
                    local_cluster_to_local(s.get(), buffer.data() + MasterOffset_output[2 * rank + 1] * j, f_local.data() + MasterOffset_output[2 * rank + 1] * j);
                } else {
                    local_cluster_to_local(t.get(), buffer.data() + MasterOffset_output[2 * rank + 1] * j, f_local.data() + MasterOffset_output[2 * rank + 1] * j);
                }
            }
        }

        // Error
        double global_local_diff = 0;
        for (int i = 0; i < MasterOffset_output[2 * rank + 1]; i++) {
            for (int j = 0; j < mu; j++) {
                global_local_diff += std::abs(f_global_test[i + MasterOffset_output[2 * rank] + j * no] - f_local[i + j * MasterOffset_output[2 * rank + 1]]) * std::abs(f_global_test[i + MasterOffset_output[2 * rank] + j * no] - f_local[i + j * MasterOffset_output[2 * rank + 1]]);
            }
        }

        double global_local_err = std::sqrt(global_local_diff) / norm2(f_global_test);

        if (rank == 0) {
            cout << "difference on mat mat prod computed globally and locally: " << global_local_err << endl;
        }
        test = test || !(global_local_err < 1e-10);
    }
    if (rank == 0) {
        cout << "test: " << test << endl;
    }

    return test;
}
