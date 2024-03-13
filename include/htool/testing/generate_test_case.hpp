
#ifndef HTOOL_TESTING_GENERATE_TEST_CASE_HPP
#define HTOOL_TESTING_GENERATE_TEST_CASE_HPP
#include "../clustering/cluster_node.hpp"
#include "../clustering/tree_builder/tree_builder.hpp"
#include "geometry.hpp"
#include "partition.hpp"
namespace htool {

template <typename T, typename GeneratorTestType>
class TestCase {
  public:
    char transa;
    char transb;
    char side;
    char symmetry;
    char UPLO;
    std::vector<underlying_type<T>> x1;
    std::vector<underlying_type<T>> x2;
    std::vector<underlying_type<T>> x3;
    std::shared_ptr<Cluster<underlying_type<T>>> root_cluster_1 = nullptr;
    std::shared_ptr<Cluster<underlying_type<T>>> root_cluster_2 = nullptr;
    std::shared_ptr<Cluster<underlying_type<T>>> root_cluster_3 = nullptr;
    const Cluster<underlying_type<T>> *root_cluster_A_input     = nullptr;
    const Cluster<underlying_type<T>> *root_cluster_A_output    = nullptr;
    const Cluster<underlying_type<T>> *root_cluster_B_input     = nullptr;
    const Cluster<underlying_type<T>> *root_cluster_B_output    = nullptr;
    const Cluster<underlying_type<T>> *root_cluster_C_input     = nullptr;
    const Cluster<underlying_type<T>> *root_cluster_C_output    = nullptr;
    std::unique_ptr<GeneratorTestType> operator_A               = nullptr;
    std::unique_ptr<GeneratorTestType> operator_B               = nullptr;
    std::unique_ptr<GeneratorTestType> operator_C               = nullptr;
    int ni_A;
    int no_A;
    int ni_B;
    int no_B;
    int ni_C;
    int no_C;

    TestCase(char transa0, char transb0, int n1, int n2, int n3, underlying_type<T> z_distance_A, underlying_type<T> z_distance_B, char side0, char symmetry0, char UPLO0, int number_of_partition = -1) : transa(transa0), transb(transb0), side(side0), symmetry(symmetry0), UPLO(UPLO0), x1(3 * n1), x2(3 * n2), x3(3 * n3) {

        // Input sizes
        ni_A = (transa == 'T' || transa == 'C') ? n1 : n2;
        no_A = (transa == 'T' || transa == 'C') ? n2 : n1;
        ni_B = (transb == 'T' || transb == 'C') ? n2 : n3;
        no_B = (transb == 'T' || transb == 'C') ? n3 : n2;
        ni_C = n3;
        no_C = n1;

        ClusterTreeBuilder<underlying_type<T>> recursive_build_strategy;

        // First geometry
        create_disk(3, 0., n1, x1.data());
        if (number_of_partition > 0) {
            std::vector<int> partition;
            test_partition(3, n1, x1, number_of_partition, partition);
            root_cluster_1 = std::make_shared<Cluster<underlying_type<T>>>(recursive_build_strategy.create_cluster_tree(n1, 3, x1.data(), 2, number_of_partition, partition.data()));
        } else {
            root_cluster_1 = std::make_shared<Cluster<underlying_type<T>>>(recursive_build_strategy.create_cluster_tree(n1, 3, x1.data(), 2, 2));
        }

        // Second geometry
        if (symmetry == 'N' || n1 != n2 || side != 'L') {
            create_disk(3, z_distance_A, n2, x2.data());
            if (number_of_partition > 0) {
                std::vector<int> partition;
                test_partition(3, n2, x2, number_of_partition, partition);
                root_cluster_2 = std::make_shared<Cluster<underlying_type<T>>>(recursive_build_strategy.create_cluster_tree(n2, 3, x2.data(), 2, number_of_partition, partition.data()));
            } else {

                root_cluster_2 = std::make_shared<Cluster<underlying_type<T>>>(recursive_build_strategy.create_cluster_tree(n2, 3, x2.data(), 2, 2));
            }
        } else {
            x2             = x1;
            root_cluster_2 = root_cluster_1;
        }

        // Third geometry
        if (symmetry == 'N' || n2 != n3 || side != 'R') {
            create_disk(3, z_distance_B, n3, x3.data());
            if (number_of_partition > 0) {
                std::vector<int> partition;
                test_partition(3, n3, x3, number_of_partition, partition);
                root_cluster_3 = std::make_shared<Cluster<underlying_type<T>>>(recursive_build_strategy.create_cluster_tree(n3, 3, x3.data(), 2, number_of_partition, partition.data()));
            } else {
                root_cluster_3 = std::make_shared<Cluster<underlying_type<T>>>(recursive_build_strategy.create_cluster_tree(n3, 3, x3.data(), 2, 2));
            }
        } else {
            x3             = x2;
            root_cluster_3 = root_cluster_2;
        }

        // Operators
        if (transa == 'N') {
            operator_A            = std::make_unique<GeneratorTestType>(3, no_A, ni_A, x1, x2, *root_cluster_1, *root_cluster_2, true, true);
            root_cluster_A_output = root_cluster_1.get();
            root_cluster_A_input  = root_cluster_2.get();
        } else {
            operator_A            = std::make_unique<GeneratorTestType>(3, no_A, ni_A, x2, x1, *root_cluster_2, *root_cluster_1, true, true);
            root_cluster_A_output = root_cluster_2.get();
            root_cluster_A_input  = root_cluster_1.get();
        }
        if (transb == 'N') {
            operator_B            = std::make_unique<GeneratorTestType>(3, no_B, ni_B, x2, x3, *root_cluster_2, *root_cluster_3, true, true);
            root_cluster_B_output = root_cluster_2.get();
            root_cluster_B_input  = root_cluster_3.get();
        } else {
            operator_B            = std::make_unique<GeneratorTestType>(3, no_B, ni_B, x3, x2, *root_cluster_3, *root_cluster_2, true, true);
            root_cluster_B_output = root_cluster_3.get();
            root_cluster_B_input  = root_cluster_2.get();
        }
        operator_C            = std::make_unique<GeneratorTestType>(3, no_C, ni_C, x1, x3, *root_cluster_1, *root_cluster_3, true, true);
        root_cluster_C_input  = root_cluster_3.get();
        root_cluster_C_output = root_cluster_1.get();
    }
};
} // namespace htool
#endif
