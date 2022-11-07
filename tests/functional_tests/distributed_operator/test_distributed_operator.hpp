#include <htool/clustering/cluster.hpp>
#include <htool/clustering/pca.hpp>
#include <htool/clustering/splitting.hpp>
#include <htool/distributed_operator/distributed_operator.hpp>
#include <htool/testing/generator_input.hpp>
#include <htool/testing/generator_test.hpp>
#include <htool/testing/geometry.hpp>
#include <htool/testing/local_dense_matrix.hpp>
#include <random>

using namespace std;
using namespace htool;

template <typename T, typename GeneratorTestType, typename ClusterImpl>
int test_distributed_operator(int nr, int nc, int mu, bool use_permutation, char Symmetry, char UPLO, char op, bool off_diagonal_approximation) {

    // Get the number of processes
    int sizeWorld;
    MPI_Comm_size(MPI_COMM_WORLD, &sizeWorld);

    // Get the rankWorld of the process
    int rankWorld;
    MPI_Comm_rank(MPI_COMM_WORLD, &rankWorld);

    srand(1);
    bool test = 0;

    // Geometry
    double z1 = 1;
    vector<double> p1(3 * nr), p1_permuted, off_diagonal_p1;
    vector<double> p2(Symmetry == 'N' ? 3 * nc : 1), p2_permuted, off_diagonal_p2;
    create_disk(3, z1, nr, p1.data());
    int size_numbering = nr / (sizeWorld);
    int count_size     = 0;
    std::vector<int> MasterOffset_target, MasterOffset_source;
    for (int p = 0; p < sizeWorld - 1; p++) {
        MasterOffset_target.push_back(count_size);
        MasterOffset_target.push_back(size_numbering);

        count_size += size_numbering;
    }
    MasterOffset_target.push_back(count_size);
    MasterOffset_target.push_back(nr - count_size);

    size_numbering = nc / sizeWorld;
    count_size     = 0;

    std::shared_ptr<VirtualCluster> cluster_source;
    std::shared_ptr<VirtualCluster> cluster_target = make_shared<Cluster<PCA<SplittingTypes::GeometricSplitting>>>();
    cluster_target->build(nr, p1.data(), MasterOffset_target.data(), 2);

    if (Symmetry == 'N') {
        double z2 = 1 + 0.1;
        create_disk(3, z2, nc, p2.data());

        for (int p = 0; p < sizeWorld - 1; p++) {
            MasterOffset_source.push_back(count_size);
            MasterOffset_source.push_back(size_numbering);

            count_size += size_numbering;
        }
        MasterOffset_source.push_back(count_size);
        MasterOffset_source.push_back(nc - count_size);
        cluster_source = make_shared<Cluster<PCA<SplittingTypes::GeometricSplitting>>>();
        cluster_source->build(nc, p2.data(), MasterOffset_source.data(), 2);
    } else {

        MasterOffset_source = MasterOffset_target;

        cluster_source = cluster_target;
        p2             = p1;
    }

    // Permutation on geometry
    if (!use_permutation || off_diagonal_approximation) {
        p1_permuted.resize(3 * nr);
        for (int i = 0; i < cluster_target->get_global_perm().size(); i++) {
            p1_permuted[i * 3 + 0] = p1[cluster_target->get_global_perm(i) * 3 + 0];
            p1_permuted[i * 3 + 1] = p1[cluster_target->get_global_perm(i) * 3 + 1];
            p1_permuted[i * 3 + 2] = p1[cluster_target->get_global_perm(i) * 3 + 2];
        }
        p2_permuted.resize(3 * nc);
        if (Symmetry == 'N') {
            for (int i = 0; i < cluster_source->get_global_perm().size(); i++) {
                p2_permuted[i * 3 + 0] = p2[cluster_source->get_global_perm(i) * 3 + 0];
                p2_permuted[i * 3 + 1] = p2[cluster_source->get_global_perm(i) * 3 + 1];
                p2_permuted[i * 3 + 2] = p2[cluster_source->get_global_perm(i) * 3 + 2];
            }
        } else {
            p2_permuted = p1_permuted;
        }
    }

    // Generator
    GeneratorTestType generator(3, nr, nc, p1, p2);
    GeneratorTestType generator_permuted(3, nr, nc, p1_permuted, p2_permuted);

    // Diagonal LocalDenseMatrix
    std::shared_ptr<LocalDenseMatrix<T>> local_dense_matrix;

    if (!off_diagonal_approximation && !use_permutation) {
        char symmetry      = (sizeWorld == 1) ? Symmetry : 'N';
        char uplo          = (sizeWorld == 1) ? UPLO : 'N';
        local_dense_matrix = make_shared<LocalDenseMatrix<T>>(generator_permuted, cluster_target->get_local_cluster_tree(), cluster_source, symmetry, uplo, false, false);
    } else if (!off_diagonal_approximation && use_permutation) {
        char symmetry      = (sizeWorld == 1) ? Symmetry : 'N';
        char uplo          = (sizeWorld == 1) ? UPLO : 'N';
        local_dense_matrix = make_shared<LocalDenseMatrix<T>>(generator, cluster_target->get_local_cluster_tree(), cluster_source, symmetry, uplo, true, true);
    } else if (off_diagonal_approximation && !use_permutation) {
        local_dense_matrix = make_shared<LocalDenseMatrix<T>>(generator_permuted, cluster_target->get_local_cluster_tree(), cluster_source->get_local_cluster_tree(), Symmetry, UPLO, false, false);
    } else {
        local_dense_matrix = make_shared<LocalDenseMatrix<T>>(generator, cluster_target->get_local_cluster_tree(), cluster_source->get_local_cluster_tree(), Symmetry, UPLO, true, true);
    }

    // Distributed operator
    DistributedOperator<T> distributed_operator(cluster_target, cluster_source, Symmetry, UPLO);
    distributed_operator.add_local_operator(local_dense_matrix);
    distributed_operator.use_permutation() = use_permutation;

    // Off diagonal geometries
    if (off_diagonal_approximation) {

        // Sizes
        int nc_local = cluster_source->get_local_size();
        int off_diagonal_nc_1{cluster_source->get_local_offset()};
        int off_diagonal_nc_2{cluster_source->get_size() - cluster_source->get_local_size() - cluster_source->get_local_offset()};

        // Local off diagonal cluster
        std::vector<int> off_diagonal_partition;
        off_diagonal_partition.push_back(0);
        off_diagonal_partition.push_back(off_diagonal_nc_1);
        // off_diagonal_partition.push_back(off_diagonal_nc_1);
        // off_diagonal_partition.push_back(nr);
        off_diagonal_partition.push_back(off_diagonal_nc_1 + nc_local);
        off_diagonal_partition.push_back(off_diagonal_nc_2);

        std::shared_ptr<Cluster<PCA<SplittingTypes::GeometricSplitting>>>
            off_diagonal_cluster = make_shared<Cluster<PCA<SplittingTypes::GeometricSplitting>>>();

        off_diagonal_cluster->build_local(nc, p2_permuted.data(), 2, off_diagonal_partition.data(), 2);

        // Generators
        std::unique_ptr<GeneratorTestType> generator_off_diagonal;

        if (use_permutation) {
            generator_off_diagonal = std::unique_ptr<GeneratorTestType>(new GeneratorTestType(3, nr, nc, p1, p2_permuted));
        } else {
            generator_off_diagonal = std::unique_ptr<GeneratorTestType>(new GeneratorTestType(3, nr, nc, p1_permuted, p2_permuted));
        }

        // Off diagonal LocalDenseMatrix
        std::shared_ptr<LocalDenseMatrix<T>> local_off_diagonal_dense_matrix_1;

        local_off_diagonal_dense_matrix_1 = make_shared<LocalDenseMatrix<T>>(*generator_off_diagonal, cluster_target->get_local_cluster_tree(), off_diagonal_cluster->get_son(0).get_cluster_tree(), 'N', 'N', use_permutation, true, false, true);

        std::shared_ptr<LocalDenseMatrix<T>> local_off_diagonal_dense_matrix_2;
        local_off_diagonal_dense_matrix_2 = make_shared<LocalDenseMatrix<T>>(*generator_off_diagonal, cluster_target->get_local_cluster_tree(), off_diagonal_cluster->get_son(1).get_cluster_tree(), 'N', 'N', use_permutation, true, false, true);

        // Add to distributed operator
        if (off_diagonal_nc_1 != 0) {
            distributed_operator.add_local_operator(local_off_diagonal_dense_matrix_1);
        }
        if (off_diagonal_nc_2 != 0) {
            distributed_operator.add_local_operator(local_off_diagonal_dense_matrix_2);
        }
    }

    // Input sizes
    int ni = (op == 'T' || op == 'C') ? nr : nc;
    int no = (op == 'T' || op == 'C') ? nc : nr;

    std::vector<int> MasterOffset_input  = (op == 'T' || op == 'C') ? MasterOffset_target : MasterOffset_source;
    std::vector<int> MasterOffset_output = (op == 'T' || op == 'C') ? MasterOffset_source : MasterOffset_target;

    // Random input vector
    std::vector<T> in_global(ni * mu, 1), random_vector(ni * mu, 1);
    if (rankWorld == 0) {
        generate_random_vector(random_vector);
    }
    MPI_Bcast(random_vector.data(), random_vector.size(), wrapper_mpi<T>::mpi_type(), 0, MPI_COMM_WORLD);

    if (use_permutation) {
        in_global = random_vector;
    } else {
        for (int j = 0; j < mu; j++) {
            if (op == 'T' || op == 'C') {
                global_to_cluster(cluster_target.get(), random_vector.data() + ni * j, in_global.data() + ni * j);
            } else {
                global_to_cluster(cluster_source.get(), random_vector.data() + ni * j, in_global.data() + ni * j);
            }
        }
    }

    // Global output vectors
    std::vector<T> out_global(no * mu, 1);
    std::vector<T> out_global_permuted(no * mu, 1);
    std::vector<T> out_ref(no * mu, 1);
    if (op == 'T') {
        generator.mvprod_transp(random_vector.data(), out_ref.data(), mu);
    } else if (op == 'C') {
        // A.mvprod_conj(random_vector.data(), f_global.data(), mu);
    } else {
        generator.mvprod(random_vector.data(), out_ref.data(), mu);
    }

    // Global product
    if (op == 'T') {
        if (mu == 1) {
            distributed_operator.vector_product_transp_global_to_global(in_global.data(), out_global.data());
        } else {
            distributed_operator.matrix_product_transp_global_to_global(in_global.data(), out_global.data(), mu);
        }
    } else if (op == 'C') {
    } else {
        if (mu == 1) {
            distributed_operator.vector_product_global_to_global(in_global.data(), out_global.data());
        } else {
            distributed_operator.matrix_product_global_to_global(in_global.data(), out_global.data(), mu);
        }
    }

    // Error on global product
    double global_error, norm_ref(norm2(out_ref));
    if (use_permutation)
        global_error = norm2(out_global - out_ref) / norm_ref;
    else {
        for (int j = 0; j < mu; j++) {
            if (op == 'T' || op == 'C') {
                cluster_to_global(cluster_source.get(), out_global.data() + no * j, out_global_permuted.data() + no * j);
            } else {
                cluster_to_global(cluster_target.get(), out_global.data() + no * j, out_global_permuted.data() + no * j);
            }
        }
        global_error = norm2(out_global_permuted - out_ref) / norm_ref;
    }
    if (rankWorld == 0)
        cout << "error with global product: " << global_error << endl;
    test = test || !(global_error < 1e-14);

    // Local vectors
    std::vector<T> x_local(MasterOffset_input[2 * rankWorld + 1] * mu), out_local(MasterOffset_output[2 * rankWorld + 1] * mu), out_local_permuted(MasterOffset_output[2 * rankWorld + 1] * mu);
    for (int i = 0; i < mu; i++) {
        std::copy_n(in_global.data() + MasterOffset_input[2 * rankWorld] + ni * i, MasterOffset_input[2 * rankWorld + 1], x_local.data() + MasterOffset_input[2 * rankWorld + 1] * i);
    }

    // Local product
    if (op == 'T') {
        if (mu == 1) {
            distributed_operator.vector_product_transp_local_to_local(x_local.data(), out_local.data());
        } else {
            distributed_operator.matrix_product_transp_local_to_local(x_local.data(), out_local.data(), mu);
        }
    } else if (op == 'C') {
        // HA->mvprod_conj_local_to_local(x_local.data(), out_local.data(), mu);
    } else {
        if (mu == 1) {
            distributed_operator.vector_product_local_to_local(x_local.data(), out_local.data());
        } else {
            distributed_operator.matrix_product_local_to_local(x_local.data(), out_local.data(), mu);
        }
    }

    // Error
    double global_local_diff = 0;
    if (!use_permutation) {
        for (int j = 0; j < mu; j++) {
            if (op == 'T' || op == 'C') {
                local_cluster_to_local(cluster_source.get(), out_local.data() + MasterOffset_output[2 * rankWorld + 1] * j, out_local_permuted.data() + MasterOffset_output[2 * rankWorld + 1] * j);
            } else {
                local_cluster_to_local(cluster_target.get(), out_local.data() + MasterOffset_output[2 * rankWorld + 1] * j, out_local_permuted.data() + MasterOffset_output[2 * rankWorld + 1] * j);
            }
        }
    }
    const T *local_output  = (use_permutation) ? out_local.data() : out_local_permuted.data();
    const T *global_output = (use_permutation) ? out_global.data() : out_global_permuted.data();
    for (int i = 0; i < MasterOffset_output[2 * rankWorld + 1]; i++) {
        for (int j = 0; j < mu; j++) {
            global_local_diff += std::abs(global_output[i + MasterOffset_output[2 * rankWorld] + j * no] - local_output[i + j * MasterOffset_output[2 * rankWorld + 1]]) * std::abs(global_output[i + MasterOffset_output[2 * rankWorld] + j * no] - local_output[i + j * MasterOffset_output[2 * rankWorld + 1]]);
        }
    }

    double global_local_err = std::sqrt(global_local_diff) / norm2(out_local);

    if (rankWorld == 0) {
        cout << "error with local product: " << global_local_err << endl;
    }
    test = test || !(global_local_err < 1e-10);

    return test;
}
