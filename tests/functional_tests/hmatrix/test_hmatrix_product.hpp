#include <htool/clustering/clustering.hpp>
#include <htool/hmatrix/hmatrix.hpp>
#include <htool/hmatrix/hmatrix_output.hpp>
#include <htool/hmatrix/tree_builder/tree_builder.hpp>
#include <htool/testing/generator_input.hpp>
#include <htool/testing/generator_test.hpp>
#include <htool/testing/geometry.hpp>
#include <htool/testing/partition.hpp>

using namespace std;
using namespace htool;

template <typename T, typename GeneratorTestType>
bool test_hmatrix_product(int nr, int nc, int mu, bool use_local_cluster, char op, char Symmetry, char UPLO, htool::underlying_type<T> epsilon) {

    // Get the number of processes
    int sizeWorld;
    MPI_Comm_size(MPI_COMM_WORLD, &sizeWorld);

    // Get the rankWorld of the process
    int rankWorld;
    MPI_Comm_rank(MPI_COMM_WORLD, &rankWorld);

    srand(1);
    bool is_error = false;

    // Geometry
    double z1 = 1;
    vector<double> p1(3 * nr), p1_permuted, off_diagonal_p1;
    vector<double> p2(Symmetry == 'N' ? 3 * nc : 1), p2_permuted, off_diagonal_p2;
    create_disk(3, z1, nr, p1.data());

    // Partition
    std::vector<std::pair<int, int>> partition{};
    test_partition(3, nr, p1, sizeWorld, partition);

    // Clustering
    ClusterTreeBuilder<htool::underlying_type<T>, ComputeLargestExtent<htool::underlying_type<T>>, RegularSplitting<htool::underlying_type<T>>> target_recursive_build_strategy(nr, 3, p1.data(), 2, sizeWorld);
    target_recursive_build_strategy.set_partition(partition);
    // target_recursive_build_strategy.set_minclustersize(2);

    std::shared_ptr<Cluster<htool::underlying_type<T>>> source_root_cluster;
    std::shared_ptr<Cluster<htool::underlying_type<T>>> target_root_cluster = make_shared<Cluster<htool::underlying_type<T>>>(target_recursive_build_strategy.create_cluster_tree());

    if (Symmetry == 'N' && nr != nc) {
        // Geometry
        double z2 = 1 + 0.1;
        create_disk(3, z2, nc, p2.data());

        // partition
        test_partition(3, nc, p2, sizeWorld, partition);

        // Clustering
        ClusterTreeBuilder<htool::underlying_type<T>, ComputeLargestExtent<htool::underlying_type<T>>, RegularSplitting<htool::underlying_type<T>>> source_recursive_build_strategy(nc, 3, p2.data(), 2, sizeWorld);
        source_recursive_build_strategy.set_partition(partition);
        // source_recursive_build_strategy.set_minclustersize(2);

        source_root_cluster = make_shared<Cluster<htool::underlying_type<T>>>(source_recursive_build_strategy.create_cluster_tree());
    } else {
        source_root_cluster = target_root_cluster;
        p2                  = p1;
    }

    // Permutation on geometry
    p1_permuted.resize(3 * nr);
    const auto &target_permutation = target_root_cluster->get_permutation();
    for (int i = 0; i < target_permutation.size(); i++) {
        p1_permuted[i * 3 + 0] = p1[target_permutation[i] * 3 + 0];
        p1_permuted[i * 3 + 1] = p1[target_permutation[i] * 3 + 1];
        p1_permuted[i * 3 + 2] = p1[target_permutation[i] * 3 + 2];
    }
    p2_permuted.resize(3 * nc);
    if (Symmetry == 'N' && nr != nc) {
        const auto &source_permutation = source_root_cluster->get_permutation();
        for (int i = 0; i < source_permutation.size(); i++) {
            p2_permuted[i * 3 + 0] = p2[source_permutation[i] * 3 + 0];
            p2_permuted[i * 3 + 1] = p2[source_permutation[i] * 3 + 1];
            p2_permuted[i * 3 + 2] = p2[source_permutation[i] * 3 + 2];
        }
    } else {
        p2_permuted = p1_permuted;
    }

    // Generator
    GeneratorTestType generator(3, nr, nc, p1_permuted, p2_permuted);

    // HMatrix
    double eta = 10;

    std::unique_ptr<HMatrixTreeBuilder<T, htool::underlying_type<T>>> hmatrix_tree_builder;
    if (use_local_cluster) {
        std::shared_ptr<Cluster<htool::underlying_type<T>>> local_target_cluster = std::make_shared<Cluster<htool::underlying_type<T>>>(clone_cluster_tree_from_partition(*target_root_cluster, rankWorld));
        std::shared_ptr<Cluster<htool::underlying_type<T>>> local_source_cluster;
        if (Symmetry == 'N' && nr != nc) {
            local_source_cluster = std::make_shared<Cluster<htool::underlying_type<T>>>(clone_cluster_tree_from_partition(*source_root_cluster, rankWorld));
        } else {
            local_source_cluster = local_target_cluster;
        }
        hmatrix_tree_builder = std::unique_ptr<HMatrixTreeBuilder<T, htool::underlying_type<T>>>(new HMatrixTreeBuilder<T, htool::underlying_type<T>>(local_target_cluster, local_source_cluster, epsilon, eta, Symmetry, UPLO));
    } else {
        hmatrix_tree_builder = std::unique_ptr<HMatrixTreeBuilder<T, htool::underlying_type<T>>>(new HMatrixTreeBuilder<T, htool::underlying_type<T>>(target_root_cluster, source_root_cluster, epsilon, eta, Symmetry, UPLO));
        hmatrix_tree_builder->set_target_partition_number(rankWorld);
    }

    // build
    HMatrix<T, htool::underlying_type<T>> root_hmatrix = hmatrix_tree_builder->build(generator);
    save_leaves_with_rank(root_hmatrix, "leaves_" + htool::NbrToStr(rankWorld));
    // Dense matrix
    const auto &hmatrix_target_cluster = root_hmatrix.get_target_cluster();
    const auto &hmatrix_source_cluster = root_hmatrix.get_source_cluster();
    std::vector<T> dense_data(hmatrix_target_cluster.get_size() * hmatrix_source_cluster.get_size());
    Matrix<T> dense_matrix;
    dense_matrix.assign(hmatrix_target_cluster.get_size(), hmatrix_source_cluster.get_size(), dense_data.data(), false);

    generator.copy_submatrix(hmatrix_target_cluster.get_size(), hmatrix_source_cluster.get_size(), hmatrix_target_cluster.get_offset(), hmatrix_source_cluster.get_offset(), dense_data.data());

    // Input
    int ni = (op == 'T' || op == 'C') ? hmatrix_target_cluster.get_size() : hmatrix_source_cluster.get_size();
    int no = (op == 'T' || op == 'C') ? hmatrix_source_cluster.get_size() : hmatrix_target_cluster.get_size();
    vector<T> x(ni * mu, 1), y(no * mu, 1), ref(no * mu, 0), out(ref);
    T alpha(1), beta(1);
    htool::underlying_type<T> error;
    generate_random_vector(x);
    generate_random_vector(y);
    generate_random_scalar(alpha);
    generate_random_scalar(beta);

    ref = y;
    dense_matrix.add_matrix_product(op, alpha, x.data(), beta, ref.data(), mu);

    // Row major inputs
    vector<T> x_row_major(ni * mu, 1), y_row_major(no * mu, 1), ref_row_major(no * mu, 0);
    for (int i = 0; i < ni; i++) {
        for (int j = 0; j < mu; j++) {
            x_row_major[i * mu + j] = x[i + j * ni];
        }
    }

    for (int i = 0; i < no; i++) {
        for (int j = 0; j < mu; j++) {
            y_row_major[i * mu + j]   = y[i + j * no];
            ref_row_major[i * mu + j] = ref[i + j * no];
        }
    }

    // Product
    if (mu == 1) {
        out = y;
        root_hmatrix.add_vector_product(op, alpha, x.data(), beta, out.data());
        error    = norm2(ref - out) / norm2(ref);
        is_error = is_error || !(error < epsilon);
        cout << "> Errors on a hmatrix vector product: " << error << endl;
    }

    out = y_row_major;
    root_hmatrix.add_matrix_product_row_major(op, alpha, x_row_major.data(), beta, out.data(), mu);
    error    = norm2(ref_row_major - out) / norm2(ref_row_major);
    is_error = is_error || !(error < epsilon);
    cout << "> Errors on a hmatrix matrix product: " << error << endl;

    // vector<T> hmatrix_to_dense(nr * nc);
    // hmatrix->copy_to_dense(hmatrix_to_dense.data());
    // Matrix<T> hmatrix_to_matrix;
    // hmatrix_to_matrix.assign(nr, nc, hmatrix_to_dense.data(), false);

    // std::cout << normFrob(hmatrix_to_matrix - dense_matrix) << "\n";
    return is_error;
}
