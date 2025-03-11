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

    // Basic tree
    HMatrix<T> &child1        = *root_hmatrix.get_children()[0].get();
    HMatrix<T> &child2        = *root_hmatrix.get_children()[1].get();
    HMatrix<T> &child1_child1 = *root_hmatrix.get_children()[0].get()->get_children()[0].get();
    HMatrix<T> &child1_child2 = *root_hmatrix.get_children()[0].get()->get_children()[1].get();

    // Tests for left_hmatrix_ancestor_of_right_hmatrix
    is_error = is_error || !(left_hmatrix_ancestor_of_right_hmatrix(root_hmatrix, child1));
    is_error = is_error || !(left_hmatrix_ancestor_of_right_hmatrix(root_hmatrix, child2));
    is_error = is_error || !(left_hmatrix_ancestor_of_right_hmatrix(root_hmatrix, child1_child1));
    is_error = is_error || !(left_hmatrix_ancestor_of_right_hmatrix(root_hmatrix, child1_child2));

    // Tests for left_hmatrix_descendant_of_right_hmatrix
    is_error = is_error || !(left_hmatrix_descendant_of_right_hmatrix(child1, root_hmatrix));
    is_error = is_error || !(left_hmatrix_descendant_of_right_hmatrix(child2, root_hmatrix));
    is_error = is_error || !(left_hmatrix_descendant_of_right_hmatrix(child1_child1, root_hmatrix));
    is_error = is_error || !(left_hmatrix_descendant_of_right_hmatrix(child1_child2, root_hmatrix));

    // Tests for enumerate_dependences
    {
        std::vector<const HMatrix<T> *> L0, dependences;

        // Test case 1: hmatrix is in L0
        {
            L0          = {&child1_child1, &child1_child2, &child2};
            dependences = enumerate_dependences<T>(child2, L0);
            is_error    = is_error || !(dependences.size() == 1);
            is_error    = is_error || !(dependences[0] == &child2);
        }

        // Test case 2: hmatrix is above L0
        {
            L0          = {&child1_child1, &child1_child2, &child2};
            dependences = enumerate_dependences<T>(root_hmatrix, L0);
            is_error    = is_error || !(dependences.size() == 3);
            is_error    = is_error || !(dependences[0] == &child1_child1);
            is_error    = is_error || !(dependences[1] == &child1_child2);
            is_error    = is_error || !(dependences[2] == &child2);
        }

        // Test case 3: hmatrix is below L0
        {
            L0 = {&root_hmatrix};

            dependences = enumerate_dependences<T>(child1, L0);
            is_error    = is_error || !(dependences.size() == 1);
            is_error    = is_error || !(dependences[0] == &root_hmatrix);

            dependences = enumerate_dependences<T>(child2, L0);
            is_error    = is_error || !(dependences.size() == 1);
            is_error    = is_error || !(dependences[0] == &root_hmatrix);

            dependences = enumerate_dependences<T>(child1_child2, L0);
            is_error    = is_error || !(dependences.size() == 1);
            is_error    = is_error || !(dependences[0] == &root_hmatrix);
        }

        // Test case 4 : L0 is empty
        // L0.clear();
        // dependences = enumerate_dependences(root_hmatrix, L0);
        // is_error    = is_error || !(dependences.empty());

    } // end of tests for enumerate_dependences

    if (is_error) {
        std::cerr << "Error: test_hmatrix_task_based failed." << std::endl;
    } else {
        std::cout << "Passed: test_hmatrix_task_based passed." << std::endl;
    }
    return is_error;
}
