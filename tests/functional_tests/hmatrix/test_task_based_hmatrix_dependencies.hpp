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
#include <htool/hmatrix/linalg/task_based_add_hmatrix_vector_product.hpp> // for task_bas...
// #include <htool/hmatrix/linalg/task_based_factorization.hpp>                    // for task_based_lu_factorization
// #include <htool/hmatrix/linalg/task_based_triangular_hmatrix_hmatrix_solve.hpp> // for task_based_triangular_hmatrix_hmatrix_solve
// #include <htool/hmatrix/linalg/triangular_hmatrix_hmatrix_solve.hpp>            // for triangular_hmatrix_hmatrix_solve

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
        std::cout << "ERROR" << std::endl;
    } else {
        std::cout << "SUCCESS" << std::endl;
    }
    std::cout << "----------------------------------" << std::endl;

    // Tests for left_hmatrix_descendant_of_right_hmatrix
    std::cout << "left_hmatrix_descendant_of_right_hmatrix tests...";
    is_error = is_error || !(left_hmatrix_descendant_of_right_hmatrix(child1, root_hmatrix));
    is_error = is_error || !(left_hmatrix_descendant_of_right_hmatrix(child2, root_hmatrix));
    is_error = is_error || !(left_hmatrix_descendant_of_right_hmatrix(child1_child1, root_hmatrix));
    is_error = is_error || !(left_hmatrix_descendant_of_right_hmatrix(child1_child2, root_hmatrix));
    if (is_error) {
        std::cout << "ERROR" << std::endl;
    } else {
        std::cout << "SUCCESS" << std::endl;
    }
    std::cout << "----------------------------------" << std::endl;

    // Tests for enumerate_dependences
    {
        std::cout << "enumerate_dependences tests...";
        std::vector<HMatrix<T> *> L0;
        std::vector<HMatrix<T> *> dependences;

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
            std::cout << "ERROR" << std::endl;
        } else {
            std::cout << "SUCCESS" << std::endl;
        }
        std::cout << "----------------------------------" << std::endl;
    } // end of tests for enumerate_dependences

    // Tests for find_l0. Visual verification of the block tree in dotfile.dot
    {
        std::vector<HMatrix<T> *> L0 = find_l0(root_hmatrix, 256);
        std::ofstream dotfile("test_find_l0.dot");
        view_block_tree(root_hmatrix, L0, dotfile);
    }

    return is_error;
}
