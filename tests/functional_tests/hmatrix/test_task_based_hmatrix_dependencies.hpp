#include <cmath>                             // for pow, sqrt
#include <cstddef>                           // for size_t
#include <htool/basic_types/vector.hpp>      // for norm2
#include <htool/clustering/cluster_node.hpp> // for Cluster...
#include <htool/clustering/cluster_output.hpp>
#include <htool/clustering/tree_builder/tree_builder.hpp> // for Cluster...
#include <htool/hmatrix/hmatrix.hpp>                      // for copy_di...
#include <htool/hmatrix/hmatrix_distributed_output.hpp>   // for print_d...
#include <htool/hmatrix/hmatrix_output.hpp>               // for print_h...
#include <htool/hmatrix/hmatrix_output_dot.hpp>           // for view_block_tree...
#include <htool/hmatrix/interfaces/virtual_generator.hpp> // for Generat...
// #include <htool/hmatrix/linalg/add_hmatrix_hmatrix_product.hpp> // for add_...
#include <htool/hmatrix/linalg/add_hmatrix_vector_product.hpp>
// #include <htool/hmatrix/linalg/factorization.hpp>
// #include <htool/hmatrix/linalg/task_based_add_hmatrix_hmatrix_product.hpp>      // for task_bas...
#include <htool/hmatrix/linalg/omp_task/add_hmatrix_vector_product.hpp> // for task_bas...
// #include <htool/hmatrix/linalg/task_based_factorization.hpp>                    // for task_based_lu_factorization
#include <htool/hmatrix/linalg/omp_task/apply_ldlt_diagonal.hpp> // for task_based_internal_apply_ldlt_diagonal
#include <htool/matrix/linalg/factorization.hpp>                 // for ldlt_factorization

#include <htool/hmatrix/lrmat/SVD.hpp>
#include <htool/hmatrix/task_dependencies.hpp>         // for enumerate_dependence, find_l0...
#include <htool/hmatrix/tree_builder/tree_builder.hpp> // for HMatrix...
#include <htool/matrix/matrix.hpp>                     // for Matrix
#include <htool/misc/misc.hpp>                         // for underly...
#include <htool/misc/user.hpp>                         // for NbrToStr
#include <htool/testing/dense_blocks_generator_test.hpp>
#include <htool/testing/generate_test_case.hpp> // for TestCaseSymmetricPro...
#include <htool/testing/generator_input.hpp>
#include <htool/testing/geometry.hpp>  // for create_...
#include <htool/testing/partition.hpp> // for test_pa...
#include <iostream>                    // for operator<<
#include <memory>                      // for make_sh...
#include <string>                      // for operator+
#include <vector>                      // for vector

using namespace std;
using namespace htool;

template <typename T, typename GeneratorTestType, typename TestCaseType>
bool test_task_based_hmatrix_dependencies(const TestCaseType &test_case, char sym, htool::underlying_type<T> epsilon, bool block_tree_consistency, char UPLO = 'L') {
    // custom tests to run
    double eta    = 10;
    bool is_error = false;

    // Get test case parameters
    const Cluster<htool::underlying_type<T>> *root_cluster_A_output, *root_cluster_A_input;
    root_cluster_A_output = test_case.root_cluster_A_output;
    root_cluster_A_input  = test_case.root_cluster_A_input;

    // Tree builder
    std::unique_ptr<HMatrixTreeBuilder<T, htool::underlying_type<T>>> hmatrix_tree_builder;
    if (sym == 'N') {
        hmatrix_tree_builder = std::make_unique<HMatrixTreeBuilder<T, htool::underlying_type<T>>>(epsilon, eta, 'N', 'N');
    } else if (sym == 'S') {
        hmatrix_tree_builder = std::make_unique<HMatrixTreeBuilder<T, htool::underlying_type<T>>>(epsilon, eta, 'S', UPLO);
    }
    hmatrix_tree_builder->set_low_rank_generator(std::make_shared<SVD<T>>(*test_case.operator_A));
    hmatrix_tree_builder->set_block_tree_consistency(block_tree_consistency);

    // build
    HMatrix<T, htool::underlying_type<T>> root_hmatrix = hmatrix_tree_builder->build(*test_case.operator_A, *root_cluster_A_output, *root_cluster_A_input, -1, -1);

    // Basic sub tree
    HMatrix<T> &child1        = *root_hmatrix.get_children()[0].get();
    HMatrix<T> &child2        = *root_hmatrix.get_children()[1].get();
    HMatrix<T> &child1_child1 = *root_hmatrix.get_children()[0].get()->get_children()[0].get();
    HMatrix<T> &child1_child2 = *root_hmatrix.get_children()[0].get()->get_children()[1].get();

    // Tests for left_hmatrix_ancestor_of_right_hmatrix
    std::cout << "left_hmatrix_ancestor_of_right_hmatrix tests...";
    is_error = is_error || !(left_hmatrix_ancestor_of_right_hmatrix(root_hmatrix, child1));
    is_error = is_error || !(left_hmatrix_ancestor_of_right_hmatrix(root_hmatrix, child2));
    is_error = is_error || !(left_hmatrix_ancestor_of_right_hmatrix(root_hmatrix, child1_child1));
    is_error = is_error || !(left_hmatrix_ancestor_of_right_hmatrix(root_hmatrix, child1_child2));

    if (is_error) {
        std::cout << "ERROR" << '\n';
    } else {
        std::cout << "SUCCESS" << '\n';
    }
    std::cout << "----------------------------------" << '\n';

    // Tests for left_hmatrix_descendant_of_right_hmatrix
    std::cout << "left_hmatrix_descendant_of_right_hmatrix tests...";
    is_error = is_error || !(left_hmatrix_descendant_of_right_hmatrix(child1, root_hmatrix));
    is_error = is_error || !(left_hmatrix_descendant_of_right_hmatrix(child2, root_hmatrix));
    is_error = is_error || !(left_hmatrix_descendant_of_right_hmatrix(child1_child1, root_hmatrix));
    is_error = is_error || !(left_hmatrix_descendant_of_right_hmatrix(child1_child2, root_hmatrix));
    if (is_error) {
        std::cout << "ERROR" << '\n';
    } else {
        std::cout << "SUCCESS" << '\n';
    }
    std::cout << "----------------------------------" << '\n';

    // Tests for get_mutable_node: every node of the block tree is reached from the root, a node outside it is not
    std::cout << "get_mutable_node tests...";
    {
        std::vector<const HMatrix<T> *> nodes{&root_hmatrix};
        for (std::size_t i = 0; i < nodes.size(); i++) {
            is_error = is_error || get_mutable_node(root_hmatrix, *nodes[i]) != nodes[i];
            for (auto &child : nodes[i]->get_children()) {
                nodes.push_back(child.get());
            }
        }
        is_error = is_error || get_mutable_node(child1, child2) != nullptr;
        is_error = is_error || get_mutable_node(child2, child1_child2) != nullptr;
        is_error = is_error || get_mutable_node(child1, child1_child2) != &child1_child2;
    }
    if (is_error) {
        std::cout << "ERROR" << '\n';
    } else {
        std::cout << "SUCCESS" << '\n';
    }
    std::cout << "----------------------------------" << '\n';

    // Tests for enumerate_dependences
    {
        std::cout << "enumerate_dependences tests...";
        std::vector<const HMatrix<T> *> L0;
        std::vector<const HMatrix<T> *> dependences;

        // Test case 1: hmatrix is in L0
        {
            L0          = {&child1_child1, &child1_child2, &child2};
            dependences = enumerate_dependences<T>(child2, L0);

            is_error = is_error || !(dependences.size() == 1);
            is_error = is_error || !(dependences[0] == &child2);
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

        if (is_error) {
            std::cout << "ERROR" << '\n';
        } else {
            std::cout << "SUCCESS" << '\n';
        }
        std::cout << "----------------------------------" << '\n';
    } // end of tests for enumerate_dependences

    // Tests for find_l0. Visual verification of the block tree in dotfile.dot
    {
        std::vector<const HMatrix<T> *> L0 = find_l0(root_hmatrix, 256);
        std::ofstream dotfile("test_find_l0.dot");
        view_block_tree(root_hmatrix, L0, dotfile);
    }

    return is_error;
}

// D^-1 applied by tasks to a block B whose L0 ancestor has the same clusters (a single part in the partition): the
// ancestor cannot be reached from B, so B itself is modified.
inline bool test_task_based_apply_ldlt_diagonal_below_L0_node_with_same_clusters() {
    bool is_error = false;
    const int n   = 20;
    std::vector<double> points(3 * n);
    for (int i = 0; i < 3 * n; i++) {
        points[i] = i;
    }
    std::vector<int> partition(n, 0);
    ClusterTreeBuilder<double> cluster_builder;
    Cluster<double> cluster     = cluster_builder.create_cluster_tree_from_global_partition(n, 3, points.data(), 2, 1, partition.data());
    const Cluster<double> &part = *cluster.get_clusters_on_partition()[0];
    is_error                    = is_error || !(part.get_offset() == cluster.get_offset() && part.get_size() == cluster.get_size());

    // A = 2 I, factorized, so that D^-1 = I / 2
    HMatrix<double> A(cluster, cluster), B(cluster, cluster);
    HMatrix<double> *A_child = A.add_child(&part, &part);
    HMatrix<double> *B_child = B.add_child(&part, &part);
    auto D                   = std::make_unique<Matrix<double>>(n, n);
    auto ones                = std::make_unique<Matrix<double>>(n, n);
    for (int i = 0; i < n; i++) {
        (*D)(i, i) = 2;
        for (int j = 0; j < n; j++) {
            (*ones)(i, j) = 1;
        }
    }
    ldlt_factorization('S', 'L', *D);
    A_child->set_dense_data(std::move(D));
    B_child->set_dense_data(std::move(ones));

    std::vector<const HMatrix<double> *> L0_A{&A}, L0_B{&B};
#if defined(_OPENMP)
#    pragma omp parallel
#    pragma omp single
#endif
    {
        task_based_internal_apply_ldlt_diagonal('S', 'L', 'L', *A_child, *B_child, L0_A, L0_B);
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            is_error = is_error || (*B_child->get_dense_data())(i, j) != 0.5;
        }
    }
    std::cout << "task-based apply_ldlt_diagonal below an L0 node with the same clusters tests..." << (is_error ? "ERROR" : "SUCCESS") << '\n';
    return is_error;
}
