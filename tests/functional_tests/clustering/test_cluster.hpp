// #include <htool/clustering/bounding_box_1.hpp>
// #include <htool/clustering/pca.hpp>
#include <htool/clustering/clustering.hpp>
#include <htool/testing/generator_input.hpp>
#include <htool/testing/geometry.hpp>
#include <htool/testing/partition.hpp>
#include <htool/wrappers/wrapper_mpi.hpp>
#include <random>

using namespace std;
using namespace htool;

template <typename T, int dim, class DirectionComputetationStrategy, class SplittingStrategy>
bool test_cluster(int size, bool use_given_partition) {

    int rankWorld, sizeWorld;
    MPI_Comm_size(MPI_COMM_WORLD, &sizeWorld);
    MPI_Comm_rank(MPI_COMM_WORLD, &rankWorld);

    srand(1);
    bool is_error = false;

    vector<T> coordinates(size * dim);
    vector<T> r(size, 0);
    vector<T> g(size, 1);

    create_disk(dim, T(0.), size, coordinates.data());

    std::vector<int> partition{};

    if (use_given_partition) {
        test_partition(dim, size, coordinates, sizeWorld, partition);
    }

    std::vector<int> nb_sons_test{2, 3, 4};
    for (auto &nb_sons : nb_sons_test) {
        if (rankWorld == 0) {
            cout << "Number of sons : " << nb_sons << endl;
        }

        ClusterTreeBuilder<T> recursive_build_strategy;
        recursive_build_strategy.set_direction_computation_strategy(std::make_shared<DirectionComputetationStrategy>());
        recursive_build_strategy.set_splitting_strategy(std::make_shared<SplittingStrategy>());
        // if (use_given_partition) {
        //     recursive_build_strategy.set_partition(partition);
        // }
        recursive_build_strategy.set_minclustersize(1);
        Cluster<T> root_cluster = recursive_build_strategy.create_cluster_tree(size, dim, coordinates.data(), nb_sons, sizeWorld, (use_given_partition) ? partition.data() : nullptr);
        is_error                = is_error || !(root_cluster.is_root());

        if (rankWorld == 0)
            print(root_cluster);

        // Testing recursivity
        std::stack<const Cluster<T> *> cluster_stack;
        cluster_stack.push(&root_cluster);
        while (!cluster_stack.empty()) {
            const Cluster<T> *curr = cluster_stack.top();
            cluster_stack.pop();
            if (!curr->is_leaf()) {
                // test num inclusion
                int count = 0;
                for (auto &child : curr->get_children()) {
                    is_error = is_error || !(curr->get_offset() + count == child->get_offset());
                    count += child->get_size();
                }

                for (auto &child : curr->get_children()) {
                    cluster_stack.push(child.get());
                }
            }
        }

        // Tests root cluster
        is_error = is_error || !(root_cluster.get_size() == size);
        is_error = is_error || !(root_cluster.get_offset() == 0);

        // Test clusters on partition
        const std::vector<const Cluster<T> *> &clusters_on_partition = root_cluster.get_clusters_on_partition();

        int summed_size_of_cluster_on_partition{0};
        for (const auto &cluster_on_partition : clusters_on_partition) {
            summed_size_of_cluster_on_partition += cluster_on_partition->get_size();
            // is_error = is_error || !(cluster_on_partition->is_cluster_on_partition() == true);
        }

        is_error = is_error || !(clusters_on_partition.size() == sizeWorld);
        is_error = is_error || !(summed_size_of_cluster_on_partition == size);

        if (use_given_partition) {
            int p = 0;
            for (const auto &cluster_on_partition : clusters_on_partition) {
                is_error = is_error || !(cluster_on_partition->get_size() == partition[2 * p + 1]);
                is_error = is_error || !(cluster_on_partition->get_offset() == partition[2 * p]);
                p++;
            }
        }

        // Testing save and read root cluster
        save_cluster_tree(root_cluster, "test_save_" + NbrToStr(rankWorld) + "_" + NbrToStr(sizeWorld));
        Cluster<T> copied_cluster = read_cluster_tree<T>("test_save_" + NbrToStr(rankWorld) + "_" + NbrToStr(sizeWorld) + "_cluster_tree_properties.csv", "test_save_" + NbrToStr(rankWorld) + "_" + NbrToStr(sizeWorld) + "_cluster_tree.csv");
        save_cluster_tree(copied_cluster, "test_save_2_" + NbrToStr(rankWorld) + "_" + NbrToStr(sizeWorld));

        is_error = is_error || !(root_cluster.get_minclustersize() == copied_cluster.get_minclustersize());
        is_error = is_error || !(root_cluster.get_maximal_depth() == copied_cluster.get_maximal_depth());
        is_error = is_error || !(root_cluster.get_minimal_depth() == copied_cluster.get_minimal_depth());
        is_error = is_error || !(root_cluster.get_permutation() == copied_cluster.get_permutation());
        is_error = is_error || !(root_cluster.is_permutation_local() == copied_cluster.is_permutation_local());

        std::stack<const Cluster<T> *> s_save;
        std::stack<const Cluster<T> *> s_read;
        s_save.push(&root_cluster);
        s_read.push(&copied_cluster);
        while (!s_save.empty()) {
            const Cluster<T> *curr_1 = s_save.top();
            const Cluster<T> *curr_2 = s_read.top();
            s_save.pop();
            s_read.pop();

            is_error = is_error || !(curr_1->get_offset() == curr_2->get_offset());
            is_error = is_error || !(curr_1->get_size() == curr_2->get_size());
            is_error = is_error || !(curr_1->get_rank() == curr_2->get_rank());
            is_error = is_error || !(curr_1->get_counter() == curr_2->get_counter());
            is_error = is_error || !(curr_1->get_depth() == curr_2->get_depth());
            is_error = is_error || !((curr_1->get_radius() - curr_2->get_radius()) < static_cast<T>(1e-5));
            is_error = is_error || !(curr_1->get_center().size() == curr_2->get_center().size());
            for (int p = 0; p < curr_1->get_center().size(); p++) {
                is_error = is_error || !((curr_1->get_center()[p] - curr_2->get_center()[p]) < static_cast<T>(1e-5));
            }

            is_error = is_error || !(curr_1->is_leaf() == curr_2->is_leaf());

            if (!curr_2->is_leaf()) {
                // test num inclusion
                for (int l = 0; l < curr_2->get_children().size(); l++) {
                    s_save.push(curr_1->get_children()[l].get());
                    s_read.push(curr_2->get_children()[l].get());
                }
            }
        }

        for (int i = 0; i < sizeWorld; i++) {
            is_error = is_error || !(root_cluster.get_clusters_on_partition()[i]->get_size() == copied_cluster.get_clusters_on_partition()[i]->get_size());
            is_error = is_error || !(root_cluster.get_clusters_on_partition()[i]->get_offset() == copied_cluster.get_clusters_on_partition()[i]->get_offset());
            is_error = is_error || !(root_cluster.get_clusters_on_partition()[i]->get_rank() == copied_cluster.get_clusters_on_partition()[i]->get_rank());
        }

        is_error = is_error || !(root_cluster.get_maximal_depth() >= root_cluster.get_minimal_depth() && root_cluster.get_minimal_depth() >= 0);

        // Test saving geometry
        if (rankWorld == 0) {
            save_clustered_geometry(root_cluster, dim, coordinates.data(), "test_cluster_geometry_" + NbrToStr(nb_sons) + "_", {0, 1, 2, 3});
        }

        // Test global renumbering
        std::vector<int> random_vector(size, 1), temporary_vector(size, 1), result_vector(size, 1);
        if (rankWorld == 0) {
            generate_random_vector(random_vector);
        }
        MPI_Bcast(random_vector.data(), random_vector.size(), wrapper_mpi<int>::mpi_type(), 0, MPI_COMM_WORLD);
        global_to_root_cluster(root_cluster, random_vector.data(), temporary_vector.data());
        root_cluster_to_global(root_cluster, temporary_vector.data(), result_vector.data());
        is_error = is_error || !(random_vector == result_vector);

        user_to_cluster(root_cluster, random_vector.data(), temporary_vector.data());
        cluster_to_user(root_cluster, temporary_vector.data(), result_vector.data());
        is_error = is_error || !(random_vector == result_vector);

        // Test renumbering on partition
        if (root_cluster.is_permutation_local()) {
            int partition_index = 0;
            for (const Cluster<T> *cluster_on_partition : clusters_on_partition) {
                std::vector<int> local_random_vector(cluster_on_partition->get_size(), 1), local_temporary_vector(cluster_on_partition->get_size(), 1), local_result_vector(cluster_on_partition->get_size(), 1);
                if (rankWorld == 0) {
                    generate_random_vector(local_random_vector);
                }
                MPI_Bcast(local_random_vector.data(), local_random_vector.size(), wrapper_mpi<int>::mpi_type(), 0, MPI_COMM_WORLD);
                local_to_local_cluster(root_cluster, partition_index, random_vector.data(), temporary_vector.data());
                local_cluster_to_local(root_cluster, partition_index, temporary_vector.data(), result_vector.data());
                is_error = is_error || !(random_vector == result_vector);

                user_to_cluster(*cluster_on_partition, random_vector.data(), temporary_vector.data());
                cluster_to_user(*cluster_on_partition, temporary_vector.data(), result_vector.data());
                is_error = is_error || !(random_vector == result_vector);
            }
            partition_index++;
        }

        if (root_cluster.is_permutation_local()) {
            // Test copy branch with local cluster
            const Cluster<T> &local_cluster = root_cluster.get_cluster_on_partition(rankWorld);
            std::stack<const Cluster<T> *> s_local_1;
            std::stack<const Cluster<T> *> s_local_2;
            s_local_1.push(&local_cluster);
            s_local_2.push(root_cluster.get_clusters_on_partition()[rankWorld]);
            while (!s_local_1.empty()) {
                const Cluster<T> *curr_1 = s_local_1.top();
                const Cluster<T> *curr_2 = s_local_2.top();
                s_local_1.pop();
                s_local_2.pop();

                is_error = is_error || !(curr_1->get_offset() == curr_2->get_offset());
                is_error = is_error || !(curr_1->get_size() == curr_2->get_size());

                if (!curr_2->is_leaf()) {
                    // test num inclusion

                    for (int l = 0; l < curr_2->get_children().size(); l++) {
                        s_local_1.push(curr_1->get_children()[l].get());
                        s_local_2.push(curr_2->get_children()[l].get());
                    }
                }
            }
            // Test renumbering with copied local cluster
            std::vector<int> local_random_vector(local_cluster.get_size(), 1), local_temporary_vector(local_cluster.get_size(), 1), local_result_vector(local_cluster.get_size(), 1);
            generate_random_vector(local_random_vector);
            local_to_local_cluster(local_cluster, rankWorld, local_random_vector.data(), local_temporary_vector.data());
            local_cluster_to_local(local_cluster, rankWorld, local_temporary_vector.data(), local_result_vector.data());
            is_error = is_error || !(local_random_vector == local_result_vector);

            user_to_cluster(local_cluster, local_random_vector.data(), local_temporary_vector.data());
            cluster_to_user(local_cluster, local_temporary_vector.data(), local_result_vector.data());
            is_error = is_error || !(local_random_vector == local_result_vector);
        }
    }

    // Permutation

    if (rankWorld == 0) {
        std::cout << "test global " << is_error << std::endl;
    }

    return is_error;
}
