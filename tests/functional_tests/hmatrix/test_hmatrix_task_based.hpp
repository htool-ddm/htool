#include <cmath>                                                  // for pow, sqrt
#include <cstddef>                                                // for size_t
#include <htool/basic_types/vector.hpp>                           // for norm2
#include <htool/clustering/cluster_node.hpp>                      // for Cluster...
#include <htool/clustering/tree_builder/recursive_build.hpp>      // for Cluster...
#include <htool/hmatrix/hmatrix.hpp>                              // for copy_di...
#include <htool/hmatrix/hmatrix_distributed_output.hpp>           // for print_d...
#include <htool/hmatrix/hmatrix_output.hpp>                       // for print_h...
#include <htool/hmatrix/hmatrix_output_dot.hpp>                   // for view_block_tree...
#include <htool/hmatrix/interfaces/virtual_generator.hpp>         // for Generat...
#include <htool/hmatrix/tree_builder/task_based_tree_builder.hpp> // for enumerate_dependence...
#include <htool/hmatrix/tree_builder/tree_builder.hpp>            // for HMatrix...
#include <htool/matrix/matrix.hpp>                                // for Matrix
#include <htool/misc/misc.hpp>                                    // for underly...
#include <htool/misc/user.hpp>                                    // for NbrToStr
#include <htool/testing/dense_blocks_generator_test.hpp>
#include <htool/testing/geometry.hpp>  // for create_...
#include <htool/testing/partition.hpp> // for test_pa...
#include <iostream>                    // for operator<<
#include <memory>                      // for make_sh...
#include <string>                      // for operator+
#include <vector>                      // for vector

using namespace std;
using namespace htool;

template <typename T, typename GeneratorTestTypeInUserNumbering>
bool test_hmatrix_task_based(int nr, int nc, char Symmetry, char UPLO, htool::underlying_type<T> epsilon) {
    bool is_error = false;

    // Geometry
    double z1 = 1;
    vector<double> p1(3 * nr), p1_permuted, off_diagonal_p1;
    vector<double> p2(Symmetry == 'N' ? 3 * nc : 1), p2_permuted, off_diagonal_p2;
    create_disk(3, z1, nr, p1.data());

    // Clustering
    ClusterTreeBuilder<htool::underlying_type<T>> cluster_tree_builder;
    // recursive_build_strategy.set_partition(partition);
    // recursive_build_strategy.set_minclustersize(2);

    std::shared_ptr<const Cluster<htool::underlying_type<T>>> source_root_cluster;
    std::shared_ptr<const Cluster<htool::underlying_type<T>>> target_root_cluster = make_shared<const Cluster<htool::underlying_type<T>>>(cluster_tree_builder.create_cluster_tree(nr, 3, p1.data(), 2, 2));

    if (Symmetry == 'N' && nr != nc) {
        // Geometry
        double z2 = 1 + 0.1;
        create_disk(3, z2, nc, p2.data());

        // Clustering
        // source_recursive_build_strategy.set_minclustersize(2);

        source_root_cluster = make_shared<const Cluster<htool::underlying_type<T>>>(cluster_tree_builder.create_cluster_tree(nc, 3, p2.data(), 2, 2));
    } else {
        source_root_cluster = target_root_cluster;
        p2                  = p1;
    }

    GeneratorTestTypeInUserNumbering generator(3, p1, p2);
    InternalGeneratorWithPermutation<T> generator_with_permutation(generator, target_root_cluster->get_permutation().data(), source_root_cluster->get_permutation().data());

    // HMatrix
    double eta = 10;

    HMatrixTreeBuilder<T, htool::underlying_type<T>> hmatrix_tree_builder(*target_root_cluster, *source_root_cluster, epsilon, eta, Symmetry, UPLO, -1, -1, -1);

    // build
    auto root_hmatrix = hmatrix_tree_builder.build(generator);

    // Visualization
    // print_hmatrix_information(root_hmatrix, std::cout);
    save_leaves_with_rank(root_hmatrix, "root_hmatrix_facto");

    //
    HMatrix<T> &child1       = *root_hmatrix.get_children()[0].get();
    HMatrix<T> &child2       = *root_hmatrix.get_children()[1].get();
    HMatrix<T> &child_child1 = *root_hmatrix.get_children()[0].get()->get_children()[0].get();
    HMatrix<T> &child_child2 = *root_hmatrix.get_children()[0].get()->get_children()[1].get();

    is_error = is_error || !(left_hmatrix_ancestor_of_right_hmatrix(root_hmatrix, child1));
    is_error = is_error || !(left_hmatrix_ancestor_of_right_hmatrix(root_hmatrix, child2));
    is_error = is_error || !(left_hmatrix_ancestor_of_right_hmatrix(root_hmatrix, child_child1));
    is_error = is_error || !(left_hmatrix_ancestor_of_right_hmatrix(root_hmatrix, child_child2));

    std::vector<const HMatrix<T> *> L0, dependences;
    // Test case 1: hmatrix is in L0
    {
        // 1-level check
        L0          = {&child1};
        dependences = enumerate_dependences<T>(child1, L0);
        is_error    = is_error || !(dependences.size() == 1);
        //     ASSERT(dependences[0] == &child1, "dependences[0] == &child1");
    }
    //     // 2-level check
    //     L0          = {&child1, &child_child2};
    //     dependences = enumerate_dependences(child_child2, L0);
    //     ASSERT(dependences.size() == 1, "dependences.size() == 1");
    //     ASSERT(dependences[0] == &child_child2, "dependences[0] == &child_child2");
    // }

    // // Test case 2: hmatrix is above L0
    // {
    //     L0          = {&child1, &child2};
    //     dependences = enumerate_dependences(root_hmatrix, L0);
    //     ASSERT(dependences.size() == 2, "dependences.size() == 2");
    //     ASSERT(dependences[0] == &child1, "dependences[0] == &child1");
    //     ASSERT(dependences[1] == &child2, "dependences[1] == &child2");
    // }

    // // Test case 3: hmatrix is below L0
    // // L0          = {&root_hmatrix};
    // L0          = find_l0(root_hmatrix, 1);
    // dependences = enumerate_dependences(child1, L0, root_hmatrix);
    // cout << "dependences.size(): " << dependences.size() << endl;
    // for (auto dep : dependences) {
    //     cout << "Dependance: " << dep << endl;
    // }
    // ASSERT(dependences.size() == 1, "dependences.size() == 1");
    // ASSERT(dependences[0] == &root_hmatrix, "dependences[0] == &root_hmatrix");

    // // Test case 4: hmatrix is not in the tree
    // dependences = enumerate_dependences(root_hmatrix, L0, child1);
    // ASSERT(dependences.empty(), "dependences.empty()");

    // // Test case 5: L0 is empty
    // L0.clear();
    // dependences = enumerate_dependences(root_hmatrix, L0, root_hmatrix);
    // ASSERT(dependences.empty(), "dependences.empty()");
    // } // end tests for enumerate_dependences
    // // tests
    // {

    //     // double criterion = cost_function(root_hmatrix);
    //     // cout << "criterion: " << criterion << endl;

    //     // std::vector<const HMatrix<double, double> *> L0;
    //     // int values[] = {0, 1, 3, 4, 15, 16, 57, 58, 225, 226, 227};
    //     // for (int i : values) {
    //     //     L0 = find_l0(root_hmatrix, i);
    //     //     cout << "i: " << i << endl;
    //     //     cout << "|L0|: " << L0.size() << endl;
    //     //     cout << "============" << endl;
    //     // }
    // view_block_tree(root_hmatrix, 64);

    //     // cout << "id : " << get_hmatrix_id(root_hmatrix) << endl;

    //     // tests for enumerate_dependences
    //     {
    //         // Create a simple tree structure
    //         HMatrix<double, double> &root         = root_hmatrix;
    //         HMatrix<double, double> &child1       = root_hmatrix->get_children()[0].get();
    //         HMatrix<double, double> &child2       = root_hmatrix->get_children()[1].get();
    //         HMatrix<double, double> &child_child1 = root_hmatrix->get_children()[0].get()->get_children()[0].get();
    //         HMatrix<double, double> &child_child2 = root_hmatrix->get_children()[0].get()->get_children()[1].get();

    //         std::vector<const HMatrix<double, double> *> L0 = {&child1};
    //         auto dependences                                = enumerate_dependences(child1, L0, root);

    // } // end tests

    return is_error;
}
