
#include <htool/clustering/clustering.hpp>
#include <htool/hmatrix/hmatrix.hpp>
#include <htool/hmatrix/hmatrix_distributed_output.hpp>
#include <htool/hmatrix/hmatrix_output.hpp>
#include <htool/hmatrix/tree_builder/tree_builder.hpp>
#include <htool/testing/dense_blocks_generator_test.hpp>
#include <htool/testing/generator_input.hpp>
#include <htool/testing/generator_test.hpp>
#include <htool/testing/geometry.hpp>
#include <htool/testing/partition.hpp>
#include <mpi.h>

using namespace std;
using namespace htool;

template <typename T, typename GeneratorTestType>
bool test_hmatrix_build(int nr, int nc, bool use_local_cluster, char Symmetry, char UPLO, htool::underlying_type<T> epsilon, bool use_dense_blocks_generator) {

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
    std::vector<int> partition{};
    test_partition(3, nr, p1, sizeWorld, partition);

    // Clustering
    ClusterTreeBuilder<htool::underlying_type<T>> recursive_build_strategy;
    // recursive_build_strategy.set_partition(partition);
    // recursive_build_strategy.set_minclustersize(2);

    std::shared_ptr<const Cluster<htool::underlying_type<T>>> source_root_cluster;
    std::shared_ptr<const Cluster<htool::underlying_type<T>>> target_root_cluster = make_shared<const Cluster<htool::underlying_type<T>>>(recursive_build_strategy.create_cluster_tree(nr, 3, p1.data(), 2, sizeWorld, partition.data()));

    if (Symmetry == 'N' && nr != nc) {
        // Geometry
        double z2 = 1 + 0.1;
        create_disk(3, z2, nc, p2.data());

        // partition
        test_partition(3, nc, p2, sizeWorld, partition);

        // Clustering
        // source_recursive_build_strategy.set_minclustersize(2);

        source_root_cluster = make_shared<const Cluster<htool::underlying_type<T>>>(recursive_build_strategy.create_cluster_tree(nc, 3, p2.data(), 2, sizeWorld, partition.data()));
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
    GeneratorTestType generator(3, nr, nc, p1_permuted, p2_permuted, *target_root_cluster, *source_root_cluster, false, false);

    // HMatrix
    double eta = 10;

    std::unique_ptr<HMatrixTreeBuilder<T, htool::underlying_type<T>>> hmatrix_tree_builder;
    if (use_local_cluster) {
        hmatrix_tree_builder = std::make_unique<HMatrixTreeBuilder<T, htool::underlying_type<T>>>(target_root_cluster->get_cluster_on_partition(rankWorld), source_root_cluster->get_cluster_on_partition(rankWorld), epsilon, eta, Symmetry, UPLO, -1, -1, -1);
    } else {
        hmatrix_tree_builder = std::make_unique<HMatrixTreeBuilder<T, htool::underlying_type<T>>>(*target_root_cluster, *source_root_cluster, epsilon, eta, Symmetry, UPLO, -1, rankWorld, rankWorld);
    }

    std::shared_ptr<VirtualDenseBlocksGenerator<T>> dense_blocks_generator;
    if (use_dense_blocks_generator) {
        dense_blocks_generator = std::make_shared<DenseBlocksGeneratorTest<T>>(generator);
    }
    hmatrix_tree_builder->set_dense_blocks_generator(dense_blocks_generator);

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
    const auto &diagonal_hmatrix = root_hmatrix.get_sub_hmatrix(*target_root_cluster->get_clusters_on_partition()[rankWorld], *source_root_cluster->get_clusters_on_partition()[rankWorld]);
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
    htool::underlying_type<T> frobenius_error{0};
    htool::underlying_type<T> dense_matrix_norm = normFrob(dense_matrix);

    if (Symmetry != 'N') {
        int local_target_offset = hmatrix_target_cluster.get_cluster_on_partition(rankWorld).get_offset() - hmatrix_target_cluster.get_offset();
        int local_source_offset = hmatrix_source_cluster.get_cluster_on_partition(rankWorld).get_offset() - hmatrix_source_cluster.get_offset();
        int local_target_size   = target_root_cluster->get_cluster_on_partition(rankWorld).get_size();
        int local_source_size   = source_root_cluster->get_cluster_on_partition(rankWorld).get_size();
        if (UPLO == 'L') {
            for (int i = 0; i < hmatrix_target_cluster.get_size(); i++) {
                for (int j = 0; j < hmatrix_source_cluster.get_size(); j++) {
                    if (!((local_target_offset <= i && i < local_target_offset + local_target_size) && (i + 1 - local_target_offset + local_source_offset <= j && j < local_source_offset + local_source_size))) {
                        frobenius_error += std::pow(std::abs((hmatrix_to_matrix(i, j) - dense_matrix(i, j))), 2);
                    }
                }
            }
        } else {
            for (int i = 0; i < hmatrix_target_cluster.get_size(); i++) {
                for (int j = 0; j < hmatrix_source_cluster.get_size(); j++) {
                    if (!((j + 1 - local_source_offset + local_target_offset <= i && i < local_target_offset + local_target_size) && (local_source_offset <= j && j < local_source_offset + local_source_size))) {
                        frobenius_error += std::pow(std::abs((hmatrix_to_matrix(i, j) - dense_matrix(i, j))), 2);
                    }
                }
            }
        }
        frobenius_error = std::sqrt(frobenius_error);
    } else {
        frobenius_error = normFrob(hmatrix_to_matrix - dense_matrix);
    }
    frobenius_error /= dense_matrix_norm;

    // if (rankWorld == 0) {
    //     std::cout << "converted :\n";
    //     hmatrix_to_matrix.print(std::cout, ",");
    // }

    is_error = is_error || !(frobenius_error < epsilon);
    std::cout << "Check dense conversion: " << frobenius_error << "\n";

    // Check get diagonal conversion
    if (Symmetry != 'N' || nr == nc) {
        int local_size   = diagonal_hmatrix->get_target_cluster().get_size();
        int local_offset = diagonal_hmatrix->get_target_cluster().get_offset();
        // const auto &permutation = root_hmatrix.get_target_cluster().get_permutation();
        vector<T> dense_diagonal(local_size);
        for (int i = 0; i < dense_matrix.nb_rows(); i++) {
            dense_diagonal[i] = generator.get_coef(i + local_offset, i + local_offset);
        }
        vector<T> hmatrix_diagonal_to_dense(local_size);
        copy_diagonal(*diagonal_hmatrix, hmatrix_diagonal_to_dense.data());
        htool::underlying_type<T> error_on_diagonal = norm2(hmatrix_diagonal_to_dense - dense_diagonal) / norm2(dense_diagonal);

        is_error = is_error || !(error_on_diagonal < epsilon);
        std::cout << "Check get diagonal: " << error_on_diagonal << "\n";
    }

    //
    if (rankWorld == 0) {
        print_tree_parameters(root_hmatrix, std::cout);
        print_hmatrix_information(root_hmatrix, std::cout);
    }
    print_distributed_hmatrix_information(root_hmatrix, std::cout, MPI_COMM_WORLD);

    return is_error;
}
