
#include <htool/clustering/clustering.hpp>
#include <htool/hmatrix/hmatrix.hpp>
#include <htool/hmatrix/hmatrix_output.hpp>
#include <htool/hmatrix/tree_builder/tree_builder.hpp>
#include <htool/testing/generator_input.hpp>
#include <htool/testing/generator_test.hpp>
#include <htool/testing/geometry.hpp>
#include <htool/testing/partition.hpp>
#include <mpi.h>

using namespace std;
using namespace htool;

template <typename T, typename GeneratorTestType>
bool test_hmatrix_build(int nr, int nc, bool use_local_cluster, char Symmetry, char UPLO, htool::underlying_type<T> epsilon) {

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

    std::shared_ptr<const Cluster<htool::underlying_type<T>>> source_root_cluster;
    std::shared_ptr<const Cluster<htool::underlying_type<T>>> target_root_cluster = make_shared<const Cluster<htool::underlying_type<T>>>(target_recursive_build_strategy.create_cluster_tree());

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

        source_root_cluster = make_shared<const Cluster<htool::underlying_type<T>>>(source_recursive_build_strategy.create_cluster_tree());
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
        std::shared_ptr<const Cluster<htool::underlying_type<T>>> local_target_cluster = std::make_shared<const Cluster<htool::underlying_type<T>>>(clone_cluster_tree_from_partition(*target_root_cluster, rankWorld));
        std::shared_ptr<const Cluster<htool::underlying_type<T>>> local_source_cluster;
        if (Symmetry == 'N' && nr != nc) {
            local_source_cluster = std::make_shared<const Cluster<htool::underlying_type<T>>>(clone_cluster_tree_from_partition(*source_root_cluster, rankWorld));
        } else {
            local_source_cluster = local_target_cluster;
        }
        hmatrix_tree_builder = std::unique_ptr<HMatrixTreeBuilder<T, htool::underlying_type<T>>>(new HMatrixTreeBuilder<T, htool::underlying_type<T>>(local_target_cluster, local_source_cluster, epsilon, eta, Symmetry, UPLO));
    } else {
        hmatrix_tree_builder = std::unique_ptr<HMatrixTreeBuilder<T, htool::underlying_type<T>>>(new HMatrixTreeBuilder<T, htool::underlying_type<T>>(target_root_cluster, source_root_cluster, epsilon, eta, Symmetry, UPLO));
        hmatrix_tree_builder->set_target_partition_number(rankWorld);
    }

    // build
    auto root_hmatrix = hmatrix_tree_builder->build(generator);

    save_leaves_with_rank(root_hmatrix, "leaves_" + htool::NbrToStr(rankWorld));
    save_levels(root_hmatrix, "level_" + htool::NbrToStr(rankWorld) + "_", {0, 1, 2});

    // Check integrity of root's children
    std::size_t check_children_size{0};
    if (root_hmatrix.get_children().size() > 0 && root_hmatrix.get_symmetry() == 'N') {
        auto &children = root_hmatrix.get_children();
        for (auto &child : children) {
            check_children_size += child->get_target_cluster().get_size() * child->get_source_cluster().get_size();
        }

        std::cout << "Check size integrity: " << root_hmatrix.get_target_cluster().get_size() * root_hmatrix.get_source_cluster().get_size() << " " << check_children_size << "\n";
        is_error = is_error || !(root_hmatrix.get_target_cluster().get_size() * root_hmatrix.get_source_cluster().get_size() == check_children_size);
    }

    // Check integrity of diagonal block
    const auto &diagonal_hmatrix = root_hmatrix.get_diagonal_hmatrix();
    if (diagonal_hmatrix == nullptr) {
        // test = test || !(Symmetry == 'N' && nr != nc);
        std::cout << "No diagonal hmatrix\n";
    } else {
        std::cout << "Check diagonal integrity: " << rankWorld << " " << diagonal_hmatrix->get_target_cluster().get_offset() << " " << diagonal_hmatrix->get_target_cluster().get_size() << " " << diagonal_hmatrix->get_source_cluster().get_offset() << " " << diagonal_hmatrix->get_source_cluster().get_size() << "\n";

        is_error = is_error || !(diagonal_hmatrix->get_target_cluster().get_offset() == target_root_cluster->get_clusters_on_partition()[rankWorld]->get_offset());
        is_error = is_error || !(diagonal_hmatrix->get_source_cluster().get_offset() == source_root_cluster->get_clusters_on_partition()[rankWorld]->get_offset());
        is_error = is_error || !(diagonal_hmatrix->get_target_cluster().get_size() == target_root_cluster->get_clusters_on_partition()[rankWorld]->get_size());
        is_error = is_error || !(diagonal_hmatrix->get_source_cluster().get_size() == source_root_cluster->get_clusters_on_partition()[rankWorld]->get_size());
    }

    // Dense matrix
    const auto &hmatrix_target_cluster = root_hmatrix.get_target_cluster();
    const auto &hmatrix_source_cluster = root_hmatrix.get_source_cluster();
    std::vector<T> dense_data(hmatrix_target_cluster.get_size() * hmatrix_source_cluster.get_size());
    Matrix<T> dense_matrix;
    dense_matrix.assign(hmatrix_target_cluster.get_size(), hmatrix_source_cluster.get_size(), dense_data.data(), false);

    generator.copy_submatrix(hmatrix_target_cluster.get_size(), hmatrix_source_cluster.get_size(), hmatrix_target_cluster.get_offset(), hmatrix_source_cluster.get_offset(), dense_data.data());

    // if (rankWorld == 0) {
    //     std::cout << "dense :\n";
    //     dense_matrix.print(std::cout, ",");
    // }

    // Check dense conversion
    // std::cout << hmatrix_target_cluster.get_offset() << " " << hmatrix_source_cluster.get_offset() << " " << hmatrix_target_cluster.get_size() << " " << hmatrix_source_cluster.get_size() << "\n";
    vector<T> hmatrix_to_dense(hmatrix_target_cluster.get_size() * hmatrix_source_cluster.get_size());
    copy_to_dense(root_hmatrix, hmatrix_to_dense.data());
    Matrix<T> hmatrix_to_matrix;
    hmatrix_to_matrix.assign(hmatrix_target_cluster.get_size(), hmatrix_source_cluster.get_size(), hmatrix_to_dense.data(), false);
    htool::underlying_type<T> frobenius_error = normFrob(hmatrix_to_matrix - dense_matrix) / normFrob(dense_matrix);

    // if (rankWorld == 0) {
    //     std::cout << "converted :\n";
    //     hmatrix_to_matrix.print(std::cout, ",");
    // }

    is_error = is_error || !(frobenius_error < epsilon);
    std::cout << "Check dense conversion: " << frobenius_error << "\n";

    // Check get diagonal conversion
    if (Symmetry != 'N' || nr == nc) {
        int local_size          = root_hmatrix.get_diagonal_hmatrix()->get_target_cluster().get_size();
        int local_offset        = root_hmatrix.get_diagonal_hmatrix()->get_target_cluster().get_offset();
        const auto &permutation = root_hmatrix.get_target_cluster().get_permutation();
        vector<T> dense_diagonal(local_size);
        for (int i = 0; i < dense_matrix.nb_rows(); i++) {
            dense_diagonal[i] = generator.get_coef(i + local_offset, i + local_offset);
        }
        vector<T> hmatrix_diagonal_to_dense(local_size);
        copy_diagonal(*root_hmatrix.get_diagonal_hmatrix(), hmatrix_diagonal_to_dense.data());
        htool::underlying_type<T> error_on_diagonal = norm2(hmatrix_diagonal_to_dense - dense_diagonal) / norm2(dense_diagonal);

        is_error = is_error || !(error_on_diagonal < epsilon);
        std::cout << "Check get diagonal: " << error_on_diagonal << "\n";
    }

    return is_error;
}
