#include <cmath>                             // for pow, sqrt
#include <cstddef>                           // for size_t
#include <htool/basic_types/vector.hpp>      // for norm2
#include <htool/clustering/cluster_node.hpp> // for Cluster...
#include <htool/clustering/cluster_output.hpp>
#include <htool/clustering/tree_builder/tree_builder.hpp>       // for Cluster...
#include <htool/hmatrix/hmatrix.hpp>                            // for copy_di...
#include <htool/hmatrix/hmatrix_distributed_output.hpp>         // for print_d...
#include <htool/hmatrix/hmatrix_output.hpp>                     // for print_h...
#include <htool/hmatrix/hmatrix_output_dot.hpp>                 // for view_block_tree...
#include <htool/hmatrix/interfaces/virtual_generator.hpp>       // for Generat...
#include <htool/hmatrix/linalg/add_hmatrix_hmatrix_product.hpp> // for add_...
#include <htool/hmatrix/linalg/add_hmatrix_vector_product.hpp>
#include <htool/hmatrix/linalg/factorization.hpp>
#include <htool/hmatrix/linalg/task_based_add_hmatrix_hmatrix_product.hpp>      // for task_bas...
#include <htool/hmatrix/linalg/task_based_add_hmatrix_vector_product.hpp>       // for task_bas...
#include <htool/hmatrix/linalg/task_based_factorization.hpp>                    // for task_based_lu_factorization
#include <htool/hmatrix/linalg/task_based_triangular_hmatrix_hmatrix_solve.hpp> // for task_based_triangular_hmatrix_hmatrix_solve
#include <htool/hmatrix/linalg/triangular_hmatrix_hmatrix_solve.hpp>            // for triangular_hmatrix_hmatrix_solve

#include <htool/hmatrix/lrmat/SVD.hpp>
#include <htool/hmatrix/tree_builder/task_based_tree_builder.hpp> // for enumerate_dependence, find_l0...
#include <htool/hmatrix/tree_builder/tree_builder.hpp>            // for HMatrix...
#include <htool/matrix/matrix.hpp>                                // for Matrix
#include <htool/misc/misc.hpp>                                    // for underly...
#include <htool/misc/user.hpp>                                    // for NbrToStr
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
bool test_task_based_hmatrix_vector_product(const TestCaseType &test_case, char sym, char transa, htool::underlying_type<T> epsilon, bool block_tree_consistency, char UPLO = 'L') {
    // custom tests to run
    bool all_tests                         = true;  // run all tests. Has priority over specific tests flags
    bool basic_tests                       = false; // tests left_hmatrix_ancestor_of_right_hmatrix, enumerate_dependences and find_l0
    bool compute_blocks_tests              = false;
    bool hmatrix_vector_product_tests      = false;
    bool assembly_and_vector_product_tests = false; // compares the assembly of a hmatrix + the product with a vector between conjoint and disjoint cases

    double eta = 10;
    std::cout << "eta = " << eta << std::endl;
    double error_tol = 1e-14;
    bool is_error    = false;

    // Get test case parameters
    const Cluster<htool::underlying_type<T>> *root_cluster_A_output, *root_cluster_A_input;
    root_cluster_A_output = test_case.root_cluster_A_output;
    root_cluster_A_input  = test_case.root_cluster_A_input;
    int ni_A              = test_case.ni_A;
    int no_A              = test_case.no_A;
    int no_B              = test_case.no_B;
    int no_C              = test_case.no_C;

    // Tree builder
    std::unique_ptr<HMatrixTreeBuilder<T, htool::underlying_type<T>>> hmatrix_tree_builder;
    if (sym == 'N') {
        hmatrix_tree_builder = std::make_unique<HMatrixTreeBuilder<T, htool::underlying_type<T>>>(epsilon, eta, 'N', 'N');
    } else if (sym == 'S') {
        hmatrix_tree_builder = std::make_unique<HMatrixTreeBuilder<T, htool::underlying_type<T>>>(epsilon, eta, 'S', UPLO);
    }
    hmatrix_tree_builder->set_low_rank_generator(std::make_shared<SVD<T>>(*test_case.operator_A));
    hmatrix_tree_builder->set_block_tree_consistency(block_tree_consistency);

    // swap input and output clusters if transa is not 'N'
    auto *input_cluster  = root_cluster_A_input;
    auto *output_cluster = root_cluster_A_output;
    if (transa != 'N') {
        std::swap(input_cluster, output_cluster);
    }

    // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // Basic tests
    // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if (basic_tests || all_tests) {
        // build
        HMatrix<T, htool::underlying_type<T>> root_hmatrix = hmatrix_tree_builder->build(*test_case.operator_A, *root_cluster_A_output, *root_cluster_A_input, -1, -1, false, 64);

        // Hmatrix Visualization
        save_leaves_with_rank(root_hmatrix, "root_hmatrix_facto");
        // print_hmatrix_information(root_hmatrix, std::cout);

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
    } // end of basic tests

    // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // Tests for task_based_compute_blocks
    // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if (compute_blocks_tests || all_tests) {
        std::cout << "task_based_compute_blocks tests...";
        std::chrono::steady_clock::time_point start, end;
        HMatrix<T> task_based_hmatrix(*root_cluster_A_output, *root_cluster_A_input);

        // task based build
        start = std::chrono::steady_clock::now();
#if defined(_OPENMP) && !defined(HTOOL_WITH_PYTHON_INTERFACE)
#    pragma omp parallel
#    pragma omp single
#endif
        {
            task_based_hmatrix = hmatrix_tree_builder->build(*test_case.operator_A, *root_cluster_A_output, *root_cluster_A_input, -1, -1, true, 64); // build
        }
        end                                               = std::chrono::steady_clock::now();
        std::chrono::duration<double> task_based_duration = end - start;

        task_based_hmatrix.get_hmatrix_tree_data()->m_information["Number_of_false_positive"] = NbrToStr(hmatrix_tree_builder->get_false_positive()); // when task_based_compute_blocks is used in build, the number of false positives is not set in the hmatrix tree data because the tasks need to finish before we can know the number of false positive

        // classic build
        start                                          = std::chrono::steady_clock::now();
        auto hmatrix                                   = hmatrix_tree_builder->build(*test_case.operator_A, *root_cluster_A_output, *root_cluster_A_input, -1, -1, false, 64); // when compute_blocks is used in build, the number of false positives is set in the hmatrix tree data
        end                                            = std::chrono::steady_clock::now();
        std::chrono::duration<double> classic_duration = end - start;

        // visu
        // std::cout << "hmatrix:" << std::endl;
        // print_hmatrix_information(hmatrix, std::cout);
        // std::cout << "task_based_hmatrix:" << std::endl;
        // print_hmatrix_information(task_based_hmatrix, std::cout);
        save_leaves_with_rank(task_based_hmatrix, "TB_hmatrix_build");
        save_leaves_with_rank(hmatrix, "hmatrix_build");

        // densification
        Matrix<T> densified_hmatrix(no_A, ni_A), densified_hmatrix_task_based(no_A, ni_A);
        copy_to_dense(hmatrix, densified_hmatrix.data());
        copy_to_dense(task_based_hmatrix, densified_hmatrix_task_based.data());

        // compare

        is_error = is_error || (std::isnan(normFrob(densified_hmatrix - densified_hmatrix_task_based) / normFrob(densified_hmatrix)) > error_tol);
        is_error = is_error || (hmatrix.get_hmatrix_tree_data()->m_information["Number_of_false_positive"] < task_based_hmatrix.get_hmatrix_tree_data()->m_information["Number_of_false_positive"]);

        if (is_error) {
            std::cout << "ERROR" << std::endl;
        } else {
            std::cout << "SUCCESS" << std::endl;
        }

        std::cout << "    normFrob(densified_hmatrix - densified_hmatrix_task_based)/normFrob(densified_hmatrix) = " << normFrob(densified_hmatrix - densified_hmatrix_task_based) / normFrob(densified_hmatrix) << std::endl;

        std::cout << "    hmatrix m_false_positive = " << hmatrix.get_hmatrix_tree_data()->m_information["Number_of_false_positive"] << std::endl;
        std::cout << "    task_based_hmatrix m_false_positive = " << task_based_hmatrix.get_hmatrix_tree_data()->m_information["Number_of_false_positive"] << std::endl
                  << std::endl;
        std::cout << "    classic_duration = " << classic_duration.count() << std::endl;
        std::cout << "    task_based_duration = " << task_based_duration.count() << std::endl
                  << std::endl;

        // check durations
        if (task_based_duration.count() > classic_duration.count()) {
            std::cerr << "    Careful: task_based_duration > classic_duration. Ratio TB/Classic = " << task_based_duration.count() / classic_duration.count() << std::endl;
        }

        std::cout << "----------------------------------" << std::endl;
    } // end of tests for task_based_compute_blocks

    // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // Tests for task_based_internal_add_hmatrix_vector_product
    // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if (hmatrix_vector_product_tests || all_tests) {
        std::cout << "task_based_internal_add_hmatrix_vector_product tests...";
        std::chrono::steady_clock::time_point start, end;

        // build
        auto hmatrix_classic    = hmatrix_tree_builder->build(*test_case.operator_A, *root_cluster_A_output, *root_cluster_A_input, -1, -1, false, 64);
        auto hmatrix_task_based = hmatrix_tree_builder->build(*test_case.operator_A, *root_cluster_A_output, *root_cluster_A_input, -1, -1, true, 64); // not in parallel area because it is not the focus of the test, but should nevertheless be called with the flag true in order to build m_L0 needed later when : hmatrix_tree_builder->get_L0();

        // visu
        // std::cout << "hmatrix_classic:" << std::endl;
        // print_hmatrix_information(hmatrix_classic, std::cout);
        // std::cout << "hmatrix_task_based:" << std::endl;
        // print_hmatrix_information(hmatrix_task_based, std::cout);
        save_leaves_with_rank(hmatrix_task_based, "TB_hmatrix_prod");
        save_leaves_with_rank(hmatrix_classic, "hmatrix_prod");

        // L0 definitions
        std::vector<HMatrix<T> *> L0                                   = hmatrix_tree_builder->get_L0();
        std::vector<const Cluster<htool::underlying_type<T>> *> in_L0  = find_l0(*input_cluster, 64);
        std::vector<const Cluster<htool::underlying_type<T>> *> out_L0 = find_l0(*output_cluster, 64);
        // std::ofstream dotfile("test_find_l0.dot");
        // view_block_tree(hmatrix_task_based, L0, dotfile);
        // for (auto &L0_nodes : L0) {
        //     // if (L0_nodes->get_target_cluster().get_offset() < L0_nodes->get_source_cluster().get_offset()) {

        //     std::cout << L0_nodes->get_target_cluster().get_offset() << " " << L0_nodes->get_target_cluster().get_size() << " ";
        //     std::cout << L0_nodes->get_source_cluster().get_offset() << " " << L0_nodes->get_source_cluster().get_size() << " " << L0_nodes->get_symmetry_for_leaves() << std ::endl;
        //     // }
        // }

        // Create a vector for the input and output
        std::vector<T> in(no_B), out(no_C, 11), out_task(no_C, 11);
        generate_random_vector(in);
        generate_random_vector(out);
        out_task        = out;
        T alpha         = T(3);
        T beta          = T(2);
        int nb_products = 50;

        // Perform the classic H-matrix vector product
        start = std::chrono::steady_clock::now();
        for (int i = 0; i < nb_products; i++) {
            openmp_internal_add_hmatrix_vector_product(transa, alpha, hmatrix_classic, in.data(), beta, out.data());
        }
        end                                            = std::chrono::steady_clock::now();
        std::chrono::duration<double> classic_duration = end - start;

        start = std::chrono::steady_clock::now();
#if defined(_OPENMP) && !defined(HTOOL_WITH_PYTHON_INTERFACE)
#    pragma omp parallel
#    pragma omp single
#endif
        {
            for (int i = 0; i < nb_products; i++) {
                // Perform the task-based internal add H-matrix vector product
                task_based_internal_add_hmatrix_vector_product(transa, alpha, hmatrix_task_based, in.data(), beta, out_task.data(), L0, in_L0, out_L0);
                // sequential_internal_add_hmatrix_vector_product(transa, alpha, hmatrix_task_based, in.data(), beta, out_task.data());
            }
        }
        end = std::chrono::steady_clock::now();

        std::chrono::duration<double> task_based_duration = end - start;

        // Compare the results
        is_error = is_error || (std::isnan(norm2(out - out_task) / norm2(out)) > nb_products * error_tol);
        if (is_error) {
            std::cout << "ERROR" << std::endl;
        } else {
            std::cout << "SUCCESS" << std::endl;
        }

        // std::cout << "    \n out_final = \n";
        // for (int i = 0; i < 0 + 15; i++) {
        //     std::cout << out[i] << " ";
        // }
        // std::cout << std::endl;

        // std::cout << "     \n out_task_final = \n";
        // for (int i = 0; i < 0 + 15; i++) {
        //     std::cout << out_task[i] << " ";
        // }
        // std::cout << std::endl;

        // std::cout << "    argmax(out - out_task) = " << argmax(out - out_task) << std::endl;
        // std::cout << "    (out - out_task) = " << (out - out_task) << std::endl;
        // std::cout << "    max(out - out_task)/out(argmax(out - out_task)) = " << max(out - out_task) / std::abs(out[argmax(out - out_task)]) << std::endl;
        std::cout << "    norm2(out) = " << norm2(out) << std::endl;

        std::cout
            << "    norm2(out - out_task)/norm2(out) = " << norm2(out - out_task) / norm2(out) << std::endl
            << std::endl;

        std::cout << "    classic_duration = " << classic_duration.count() << std::endl;
        std::cout << "    task_based_duration = " << task_based_duration.count() << std::endl;

        // check durations
        if (task_based_duration.count() > classic_duration.count()) {
            std::cerr << "Careful: task_based_duration > classic_duration. Ratio TB/Classic = " << task_based_duration.count() / classic_duration.count() << std::endl;
        }
        std::cout << "----------------------------------" << std::endl;
    } // end of tests for task_based_internal_add_hmatrix_vector_product

    // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // Tests for hmatrix vector product following assembly with task_based_compute_blocks and task_based_internal_add_hmatrix_vector_product
    // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if (assembly_and_vector_product_tests || all_tests) {
        std::cout << "assembly + hmatrix vector products tests...";

        int nb_products = 100;
        T alpha         = T(3);
        T beta          = T(2);
        std::chrono::duration<double> disjoint_case_duration, conjoint_case_duration;
        std::chrono::steady_clock::time_point start, end;
        HMatrix<T> task_based_hmatrix(*root_cluster_A_output, *root_cluster_A_input);

        std::vector<const Cluster<htool::underlying_type<T>> *> in_L0  = find_l0(*input_cluster, 64);
        std::vector<const Cluster<htool::underlying_type<T>> *> out_L0 = find_l0(*output_cluster, 64);
        std::vector<T> in(no_B), out_task_dis(no_C, 11), out_task_con(no_C, 11);
        generate_random_vector(in);
        generate_random_vector(out_task_dis);
        out_task_con = out_task_dis;

        // Case 1 : task graphs are disjoint
        {
            // build
            start = std::chrono::steady_clock::now();
#if defined(_OPENMP) && !defined(HTOOL_WITH_PYTHON_INTERFACE)
#    pragma omp parallel
#    pragma omp single
#endif
            {
                task_based_hmatrix = hmatrix_tree_builder->build(*test_case.operator_A, *root_cluster_A_output, *root_cluster_A_input, -1, -1, true, 64);
            }

            // N hmatrix vector products
            std::vector<HMatrix<T> *> L0 = hmatrix_tree_builder->get_L0();

#if defined(_OPENMP) && !defined(HTOOL_WITH_PYTHON_INTERFACE)
#    pragma omp parallel
#    pragma omp single
#endif
            {
                for (int i = 0; i < nb_products; i++) {
                    task_based_internal_add_hmatrix_vector_product(transa, alpha, task_based_hmatrix, in.data(), beta, out_task_dis.data(), L0, in_L0, out_L0);
                }
            }
            end                    = std::chrono::steady_clock::now();
            disjoint_case_duration = end - start;
            std::cout << "\n    disjoint_case_TB_duration = " << disjoint_case_duration.count() << std::endl;
        } // end of case 1

        // Case 2 : task graphs are not conjoint
        {
            // build
            start = std::chrono::steady_clock::now();
#if defined(_OPENMP) && !defined(HTOOL_WITH_PYTHON_INTERFACE)
#    pragma omp parallel
#    pragma omp single
#endif
            {
                task_based_hmatrix = hmatrix_tree_builder->build(*test_case.operator_A, *root_cluster_A_output, *root_cluster_A_input, -1, -1, true, 64);

                // N hmatrix vector products
                std::vector<HMatrix<T> *> L0 = hmatrix_tree_builder->get_L0();

                for (int i = 0; i < nb_products; i++) {
                    task_based_internal_add_hmatrix_vector_product(transa, alpha, task_based_hmatrix, in.data(), beta, out_task_con.data(), L0, in_L0, out_L0);
                }
            }
            end                    = std::chrono::steady_clock::now();
            conjoint_case_duration = end - start;
            std::cout << "    conjoint_case_TB_duration = " << conjoint_case_duration.count() << std::endl;

            is_error = is_error || (std::isnan(norm2(out_task_con - out_task_dis) / norm2(out_task_con)) > error_tol);
            std::cout << "    norm2(out_task_con - out_task_dis)/norm2(out_task_con) = " << norm2(out_task_con - out_task_dis) / norm2(out_task_con) << std::endl;
        } // end of case 2

        if (disjoint_case_duration.count() < conjoint_case_duration.count()) {
            std::cerr << "Careful: conjoint_case_TB_duration > disjoint_case_TB_duration. Ratio conjoint/disjoint = " << conjoint_case_duration.count() / disjoint_case_duration.count() << std::endl;
        }

        std::cout << "----------------------------------" << std::endl;
    }

    // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // Print the results
    // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if (is_error) {
        std::cerr << "ERROR: test_task_based_hmatrix_vector_product current case failed." << std::endl;
    } else {
        std::cout << "SUCCESS: test_task_based_hmatrix_vector_product current case passed." << std::endl;
        std::cout << "===============================================================\n"
                  << std::endl;
    }
    return is_error;
}

template <typename T, typename GeneratorTestType, typename TestCaseType>
bool test_task_based_hmatrix_hmatrix_product(const TestCaseType &test_case, char sym, char transa, char transb, htool::underlying_type<T> epsilon, bool block_tree_consistency, char UPLO = 'L') {
    bool is_error    = false;
    double eta       = 10;
    double error_tol = 1e-15;
    std::cout << "eta = " << eta << std::endl;
    std::cout << "task_based_internal_add_hmatrix_hmatrix_product tests...";

    // Get test case parameters
    const Cluster<htool::underlying_type<T>> *root_cluster_A_output, *root_cluster_A_input,
        *root_cluster_B_output, *root_cluster_B_input, *root_cluster_C_output, *root_cluster_C_input;
    root_cluster_A_output = test_case.root_cluster_A_output;
    root_cluster_A_input  = test_case.root_cluster_A_input;
    root_cluster_B_output = test_case.root_cluster_B_output;
    root_cluster_B_input  = test_case.root_cluster_B_input;
    root_cluster_C_output = test_case.root_cluster_C_output;
    root_cluster_C_input  = test_case.root_cluster_C_input;
    int ni_C              = test_case.ni_C;
    int no_C              = test_case.no_C;

    // Tree builder
    std::unique_ptr<HMatrixTreeBuilder<T, htool::underlying_type<T>>> hmatrix_tree_builder_A,
        hmatrix_tree_builder_B, hmatrix_tree_builder_C;
    if (sym == 'N') {
        hmatrix_tree_builder_A = std::make_unique<HMatrixTreeBuilder<T, htool::underlying_type<T>>>(epsilon, eta, 'N', 'N');
        hmatrix_tree_builder_B = std::make_unique<HMatrixTreeBuilder<T, htool::underlying_type<T>>>(epsilon, eta, 'N', 'N');
        hmatrix_tree_builder_C = std::make_unique<HMatrixTreeBuilder<T, htool::underlying_type<T>>>(epsilon, eta, 'N', 'N');
    } else if (sym == 'S') {
        hmatrix_tree_builder_A = std::make_unique<HMatrixTreeBuilder<T, htool::underlying_type<T>>>(epsilon, eta, 'S', UPLO);
        hmatrix_tree_builder_B = std::make_unique<HMatrixTreeBuilder<T, htool::underlying_type<T>>>(epsilon, eta, 'S', UPLO);
        hmatrix_tree_builder_C = std::make_unique<HMatrixTreeBuilder<T, htool::underlying_type<T>>>(epsilon, eta, 'S', UPLO);
    }

    hmatrix_tree_builder_A->set_low_rank_generator(std::make_shared<SVD<T>>(*test_case.operator_A));
    hmatrix_tree_builder_A->set_block_tree_consistency(block_tree_consistency);

    // swap input and output clusters if transX is not 'N'
    // auto *input_cluster_A  = &hmatrix_tree_builder_A->get_source_cluster();
    // auto *output_cluster_A = &hmatrix_tree_builder_A->get_target_cluster();
    // if (transa != 'N') {
    //     std::swap(input_cluster, output_cluster);
    // }
    // auto *input_cluster_B  = &hmatrix_tree_builder_B->get_source_cluster();
    // auto *output_cluster_B = &hmatrix_tree_builder_B->get_target_cluster();
    // if (transb != 'N') {
    //     std::swap(input_cluster_B, output_cluster_B);
    // }

    // build
    auto hmatrix_task_based_A      = hmatrix_tree_builder_A->build(*test_case.operator_A, *root_cluster_A_output, *root_cluster_A_input, -1, -1, true, 64);
    auto hmatrix_task_based_B      = hmatrix_tree_builder_B->build(*test_case.operator_B, *root_cluster_B_output, *root_cluster_B_input, -1, -1, true, 64);
    auto hmatrix_classic_C         = hmatrix_tree_builder_C->build(*test_case.operator_C, *root_cluster_C_output, *root_cluster_C_input, -1, -1, false, 64);
    auto hmatrix_task_based_C      = hmatrix_tree_builder_C->build(*test_case.operator_C, *root_cluster_C_output, *root_cluster_C_input, -1, -1, true, 64);
    std::vector<HMatrix<T> *> L0   = hmatrix_tree_builder_C->get_L0();
    std::vector<HMatrix<T> *> L0_A = hmatrix_tree_builder_A->get_L0();
    std::vector<HMatrix<T> *> L0_B = hmatrix_tree_builder_B->get_L0();

    // visu
    // std::cout << "hmatrix_task_based_A:" << std::endl;
    // print_hmatrix_information(hmatrix_task_based_A, std::cout);
    // std::cout << "hmatrix_task_based_B:" << std::endl;
    // print_hmatrix_information(hmatrix_task_based_B, std::cout);
    // std::cout << "hmatrix_classic_C:" << std::endl;
    // print_hmatrix_information(hmatrix_classic_C, std::cout);
    // std::cout << "hmatrix_task_based_C:" << std::endl;
    // print_hmatrix_information(hmatrix_task_based_C, std::cout);
    save_leaves_with_rank(hmatrix_task_based_A, "hmatrix_task_based_AAAAAAAAAAAAAAAAA");
    save_leaves_with_rank(hmatrix_task_based_B, "hmatrix_task_based_B");
    save_leaves_with_rank(hmatrix_classic_C, "hmatrix_classic_C");
    save_leaves_with_rank(hmatrix_task_based_C, "hmatrix_task_based_C");

    // parameters
    T alpha         = T(3);
    T beta          = T(2);
    int nb_products = 1;

    std::chrono::steady_clock::time_point start, end;

    // Perform the classic hmatrix hmatrix product
    start = std::chrono::steady_clock::now();
    for (int i = 0; i < nb_products; i++) {
        internal_add_hmatrix_hmatrix_product(transa, transb, alpha, hmatrix_task_based_A, hmatrix_task_based_B, beta, hmatrix_classic_C);
    }
    end                                            = std::chrono::steady_clock::now();
    std::chrono::duration<double> classic_duration = end - start;

    // Perform the task-based internal add H-matrix hamtrix product
    start = std::chrono::steady_clock::now();
#if defined(_OPENMP) && !defined(HTOOL_WITH_PYTHON_INTERFACE)
#    pragma omp parallel
#    pragma omp single
#endif
    {
        for (int i = 0; i < nb_products; i++) {
            task_based_internal_add_hmatrix_hmatrix_product(transa, transb, alpha, hmatrix_task_based_A, hmatrix_task_based_B, beta, hmatrix_task_based_C, L0_A, L0_B, L0);
        }
    }
    end                                               = std::chrono::steady_clock::now();
    std::chrono::duration<double> task_based_duration = end - start;

    // desify results
    Matrix<T> densified_hmatrix_classic_C(no_C, ni_C), densified_hmatrix_task_based_C(no_C, ni_C);
    copy_to_dense(hmatrix_classic_C, densified_hmatrix_classic_C.data());
    copy_to_dense(hmatrix_task_based_C, densified_hmatrix_task_based_C.data());
    // Matrix<T> test = densified_hmatrix_classic_C - densified_hmatrix_task_based_C;
    // std::ofstream dense_classic_file("dense_classic.csv");
    // std::ofstream dense_task_file("dense_task.csv");
    // std::ofstream test_file("test.csv");
    // densified_hmatrix_classic_C.print(dense_classic_file, ",");
    // densified_hmatrix_task_based_C.print(dense_task_file, ",");
    // test.print(test_file, ",");

    // Compare the results
    is_error = is_error || (std::isnan(normFrob(densified_hmatrix_classic_C - densified_hmatrix_task_based_C) / normFrob(densified_hmatrix_classic_C)) > nb_products * error_tol);

    // Print the results
    if (is_error) {
        std::cout << "ERROR" << std::endl;
    } else {
        std::cout << "SUCCESS" << std::endl;
    }

    std::cout << "    normFrob(densified_hmatrix_classic_C) = " << normFrob(densified_hmatrix_classic_C) << std::endl;
    std::cout << "    normFrob(densified_hmatrix_task_based_C) = " << normFrob(densified_hmatrix_task_based_C) << std::endl;

    std::cout << "    normFrob(classic_C - task_based_C) / normFrob(classic_C) = " << normFrob(densified_hmatrix_classic_C - densified_hmatrix_task_based_C) / normFrob(densified_hmatrix_classic_C) << std::endl
              << std::endl;

    std::cout << "    classic_duration = " << classic_duration.count() << std::endl;
    std::cout << "    task_based_duration = " << task_based_duration.count() << std::endl;

    // check durations
    if (task_based_duration.count() > classic_duration.count()) {
        std::cerr << "Careful: task_based_duration > classic_duration. Ratio TB/Classic = " << task_based_duration.count() / classic_duration.count() << std::endl;
    }
    std::cout << "----------------------------------" << std::endl;

    // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // Print the results
    // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if (is_error) {
        std::cerr << "ERROR: test_task_based_hmatrix_hmatrix_product current case failed." << std::endl;
    } else {
        std::cout << "SUCCESS: test_task_based_hmatrix_hmatrix_product current case passed." << std::endl;
        std::cout << "===============================================================\n"
                  << std::endl;
    }
    return is_error;
}

template <typename T, typename GeneratorTestType, typename TestCaseType>
bool test_task_based_hmatrix_triangular_solve(const TestCaseType &test_case, char side, char transa, char diag, double epsilon) {

    bool is_error = false;
    double eta    = 10;
    htool::underlying_type<T> error;
    std::cout << "eta = " << eta << std::endl;
    std::cout << "task_based_internal_triangular_hmatrix_hmatrix_solve tests...\n";

    // Random input
    T alpha(1);
    generate_random_scalar(alpha);

    // HMatrix
    HMatrixTreeBuilder<T, htool::underlying_type<T>> hmatrix_tree_builder_A(epsilon, eta, 'N', 'N');
    HMatrixTreeBuilder<T, htool::underlying_type<T>> hmatrix_tree_builder_X(epsilon, eta, 'N', 'N');

    HMatrix<T, htool::underlying_type<T>> A = hmatrix_tree_builder_A.build(*test_case.operator_A, *test_case.root_cluster_A_output, *test_case.root_cluster_A_input, -1, -1, true, 64);
    HMatrix<T, htool::underlying_type<T>> X = hmatrix_tree_builder_X.build(*test_case.operator_X, *test_case.root_cluster_X_output, *test_case.root_cluster_X_input, -1, -1, true, 64);
    HMatrix<T, htool::underlying_type<T>> B(X), UB(X), LB(X);
    HMatrix<T, htool::underlying_type<T>> hmatrix_test(B);

    if (diag == 'U') { // to avoid conditioning issues with the unit diagonal
        std::vector<T> diagonal(A.nb_cols());
        copy_diagonal(A, diagonal.data());
        auto max_abs = *std::max_element(diagonal.begin(), diagonal.end(), [](const T &a, const T &b) {
            return std::abs(a) < std::abs(b);
        });
        scale(T(1) / (T(10) * std::abs(max_abs)), A);
    }

    // Triangular hmatrices
    HMatrix<T, htool::underlying_type<T>> LA(A);
    HMatrix<T, htool::underlying_type<T>> UA(A);
    preorder_tree_traversal(LA, [&diag](HMatrix<T, htool::underlying_type<T>> &hmatrix) {
        if (hmatrix.is_leaf() and hmatrix.get_target_cluster() == hmatrix.get_source_cluster()) {
            Matrix<T> &dense_data = *hmatrix.get_dense_data();
            for (int i = 0; i < dense_data.nb_rows(); i++) {
                if (diag == 'U') {
                    dense_data(i, i) = 1;
                }
                for (int j = i + 1; j < dense_data.nb_cols(); j++) {
                    dense_data(i, j) = 0;
                }
            }
        } else {
            std::vector<std::unique_ptr<HMatrix<T, htool::underlying_type<T>>>> filtered_children;
            for (auto &child : hmatrix.get_children_with_ownership()) {
                if (child->get_target_cluster().get_offset() >= child->get_source_cluster().get_offset()) {
                    filtered_children.push_back(std::move(child));
                }
            }
            if (filtered_children.size() > 0) {
                hmatrix.delete_children();
                hmatrix.assign_children(filtered_children);
            }
        }
    });

    preorder_tree_traversal(UA, [&diag](HMatrix<T, htool::underlying_type<T>> &hmatrix) {
        if (hmatrix.is_leaf() and hmatrix.get_target_cluster() == hmatrix.get_source_cluster()) {
            Matrix<T> &dense_data = *hmatrix.get_dense_data();
            for (int j = 0; j < dense_data.nb_cols(); j++) {
                if (diag == 'U') {
                    dense_data(j, j) = 1;
                }
                for (int i = j + 1; i < dense_data.nb_rows(); i++) {
                    dense_data(i, j) = 0;
                }
            }
        } else {
            std::vector<std::unique_ptr<HMatrix<T, htool::underlying_type<T>>>> filtered_children;
            for (auto &child : hmatrix.get_children_with_ownership()) {
                if (child->get_target_cluster().get_offset() <= child->get_source_cluster().get_offset()) {
                    filtered_children.push_back(std::move(child));
                }
            }
            if (filtered_children.size() > 0) {
                hmatrix.delete_children();
                hmatrix.assign_children(filtered_children);
            }
        }
    });

    // Matrix
    int ni_A = test_case.root_cluster_A_input->get_size();
    int no_A = test_case.root_cluster_A_output->get_size();
    int ni_X = test_case.root_cluster_X_input->get_size();
    int no_X = test_case.root_cluster_X_output->get_size();
    Matrix<T> A_dense(no_A, ni_A), X_dense(no_X, ni_X), B_dense(X_dense), densified_hmatrix_test(B_dense), matrix_test;
    test_case.operator_A->copy_submatrix(no_A, ni_A, test_case.root_cluster_A_output->get_offset(), test_case.root_cluster_A_input->get_offset(), A_dense.data());
    test_case.operator_X->copy_submatrix(no_X, ni_X, test_case.root_cluster_X_output->get_offset(), test_case.root_cluster_X_input->get_offset(), X_dense.data());

    // Triangular matrices
    if (side == 'L') {
        internal_add_hmatrix_hmatrix_product(transa, 'N', T(1) / alpha, UA, X, T(0), UB);
        internal_add_hmatrix_hmatrix_product(transa, 'N', T(1) / alpha, LA, X, T(0), LB);
    } else {
        internal_add_hmatrix_hmatrix_product('N', transa, T(1) / alpha, X, UA, T(0), UB);
        internal_add_hmatrix_hmatrix_product('N', transa, T(1) / alpha, X, LA, T(0), LB);
    }

    // save_leaves_with_rank(LA, "LA");
    // save_leaves_with_rank(LB, "LB");

    // Tests
    std::chrono::steady_clock::time_point start, end;
    int max_nb_nodes                = 32;
    std::vector<HMatrix<T> *> L0_LA = find_l0(LA, max_nb_nodes);
    std::vector<HMatrix<T> *> L0_UA = find_l0(UA, max_nb_nodes);
    std::vector<HMatrix<T> *> L0_test;

    //// internal_triangular_hmatrix_hmatrix_solve Lower
    ////// Classic
    hmatrix_test = LB;
    start        = std::chrono::steady_clock::now();
    internal_triangular_hmatrix_hmatrix_solve(side, 'L', transa, diag, alpha, LA, hmatrix_test);
    end                                            = std::chrono::steady_clock::now();
    std::chrono::duration<double> classic_duration = end - start;
    copy_to_dense(hmatrix_test, densified_hmatrix_test.data());
    error    = normFrob(X_dense - densified_hmatrix_test) / normFrob(X_dense);
    is_error = is_error || !(error < epsilon);
    cout << ">> Lower case: " << endl;
    cout << ">   classic errors = " << error << endl;
    cout << "    classic_duration = " << classic_duration.count() << std::endl;

    ////// Task-based
    hmatrix_test = LB;
    L0_test      = find_l0(hmatrix_test, max_nb_nodes);
    start        = std::chrono::steady_clock::now();
#if defined(_OPENMP) && !defined(HTOOL_WITH_PYTHON_INTERFACE)
#    pragma omp parallel
#    pragma omp single
#endif
    {
        task_based_internal_triangular_hmatrix_hmatrix_solve(side, 'L', transa, diag, alpha, LA, hmatrix_test, L0_LA, L0_test);
    }
    end                                               = std::chrono::steady_clock::now();
    std::chrono::duration<double> task_based_duration = end - start;
    copy_to_dense(hmatrix_test, densified_hmatrix_test.data());
    error    = normFrob(X_dense - densified_hmatrix_test) / normFrob(X_dense);
    is_error = is_error || !(error < epsilon);
    cout << ">   task_based errors = " << error << endl;
    cout << "    task_based_duration = " << task_based_duration.count() << endl;
    if (task_based_duration.count() > classic_duration.count()) {
        std::cerr << "Careful: task_based_duration > classic_duration. Ratio TB/Classic = " << task_based_duration.count() / classic_duration.count() << std::endl;
    }

    // internal_triangular_hmatrix_hmatrix_solve Upper
    //// Classic
    hmatrix_test = UB;
    start        = std::chrono::steady_clock::now();
    internal_triangular_hmatrix_hmatrix_solve(side, 'U', transa, diag, alpha, UA, hmatrix_test);
    end              = std::chrono::steady_clock::now();
    classic_duration = end - start;
    copy_to_dense(hmatrix_test, densified_hmatrix_test.data());
    error    = normFrob(X_dense - densified_hmatrix_test) / normFrob(X_dense);
    is_error = is_error || !(error < epsilon);
    cout << ">> Upper case: " << endl;
    cout << ">   classic error = " << error << endl;
    cout << "    classic_duration = " << classic_duration.count() << std::endl;

    //// Task-based
    hmatrix_test = UB;
    L0_test      = find_l0(hmatrix_test, max_nb_nodes);
    start        = std::chrono::steady_clock::now();
#if defined(_OPENMP) && !defined(HTOOL_WITH_PYTHON_INTERFACE)
#    pragma omp parallel
#    pragma omp single
#endif
    {
        task_based_internal_triangular_hmatrix_hmatrix_solve(side, 'U', transa, diag, alpha, UA, hmatrix_test, L0_UA, L0_test);
    }
    end                 = std::chrono::steady_clock::now();
    task_based_duration = end - start;
    copy_to_dense(hmatrix_test, densified_hmatrix_test.data());
    error    = normFrob(X_dense - densified_hmatrix_test) / normFrob(X_dense);
    is_error = is_error || !(error < epsilon);
    cout << ">   task_based errors = " << error << endl;
    cout << "    task_based_duration = " << task_based_duration.count() << endl;
    if (task_based_duration.count() > classic_duration.count()) {
        std::cerr << "Careful: task_based_duration > classic_duration. Ratio TB/Classic = " << task_based_duration.count() / classic_duration.count() << std::endl;
    }

    // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // Print the results
    // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if (is_error) {
        std::cerr << "ERROR: test_task_based_hmatrix_triangular_solve current case failed." << std::endl;
    } else {
        std::cout << "SUCCESS: test_task_based_hmatrix_triangular_solve current case passed." << std::endl;
        std::cout << "===============================================================\n"
                  << std::endl;
    }
    return is_error;
} // end of test_task_based_hmatrix_triangular_solve

template <typename T, typename GeneratorTestType>
bool test_task_based_lu_factorization(char trans, int n1, int n2, htool::underlying_type<T> epsilon) {

    bool is_error = false;
    double eta    = 100;
    htool::underlying_type<T> error;
    std::cout << "eta = " << eta << std::endl;
    std::cout << "task_based_lu_factorization tests...\n";

    // Setup test case
    htool::TestCaseSolve<T, GeneratorTestType> test_case('L', trans, n1, n2, 1, -1);

    // HMatrix
    HMatrixTreeBuilder<T, htool::underlying_type<T>> hmatrix_tree_builder_A(epsilon, eta, 'N', 'N');
    HMatrix<T, htool::underlying_type<T>> A = hmatrix_tree_builder_A.build(*test_case.operator_A, *test_case.root_cluster_A_output, *test_case.root_cluster_A_input);

    // Matrix
    int ni_A = test_case.root_cluster_A_input->get_size();
    int no_A = test_case.root_cluster_A_output->get_size();
    int ni_X = test_case.root_cluster_X_input->get_size();
    int no_X = test_case.root_cluster_X_output->get_size();
    Matrix<T> A_dense(no_A, ni_A), X_dense(no_X, ni_X), B_dense(X_dense), densified_hmatrix_test(B_dense), matrix_test;
    std::vector<int> identity(ni_A);
    std::iota(identity.begin(), identity.end(), test_case.root_cluster_A_output->get_offset());
    test_case.operator_in_user_numbering_A->copy_submatrix(no_A, ni_A, identity.data(), identity.data(), A_dense.data());
    generate_random_matrix(X_dense);
    add_matrix_matrix_product(trans, 'N', T(1.), A_dense, X_dense, T(0.), B_dense);

    // Tests
    std::chrono::steady_clock::time_point start, end;

    //// Classic LU factorization
    auto A_classic = A;
    matrix_test    = B_dense;
    start          = std::chrono::steady_clock::now();
    lu_factorization(A_classic);
    end = std::chrono::steady_clock::now();
    // save_leaves_with_rank(A_classic, "classic_hmatrix_facto");
    lu_solve(trans, A_classic, matrix_test);
    std::chrono::duration<double> classic_duration = end - start;
    error                                          = normFrob(X_dense - matrix_test) / normFrob(X_dense);
    is_error                                       = is_error || !(error < epsilon);
    cout << ">   classic error = " << error << endl;
    cout << "    classic_duration = " << classic_duration.count() << std::endl;

    //// Task-based LU factorization
    auto A_task_based              = A;
    int max_nb_nodes               = 64;
    std::vector<HMatrix<T> *> L0_A = find_l0(A_task_based, max_nb_nodes);
    matrix_test                    = B_dense;
    start                          = std::chrono::steady_clock::now();
#if defined(_OPENMP) && !defined(HTOOL_WITH_PYTHON_INTERFACE)
#    pragma omp parallel
#    pragma omp single
#endif
    {
        task_based_lu_factorization(A_task_based, L0_A);
    }
    end = std::chrono::steady_clock::now();
    // save_leaves_with_rank(A_task_based, "TB_hmatrix_facto");
    lu_solve(trans, A_task_based, matrix_test);
    std::chrono::duration<double> task_based_duration = end - start;
    error                                             = normFrob(X_dense - matrix_test) / normFrob(X_dense);
    is_error                                          = is_error || !(error < epsilon);
    cout << ">   task_based error = " << error << endl;
    cout << "    task_based_duration = " << task_based_duration.count() << std::endl;
    if (task_based_duration.count() > classic_duration.count()) {
        std::cerr << "Careful: task_based_duration > classic_duration. Ratio TB/Classic = " << task_based_duration.count() / classic_duration.count() << std::endl;
    }

    // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // Print the results
    // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if (is_error) {
        std::cerr << "ERROR: test_task_based_lu_factorization current case failed." << std::endl;
    } else {
        std::cout << "SUCCESS: test_task_based_lu_factorization current case passed." << std::endl;
        std::cout << "===============================================================\n"
                  << std::endl;
    }
    return is_error;
}

template <typename T, typename GeneratorTestType, std::enable_if_t<!is_complex_t<T>::value, bool> = true>
bool test_task_based_cholesky_factorization(char UPLO, int n1, int n2, htool::underlying_type<T> epsilon) {
    bool is_error = false;
    double eta    = 100;
    htool::underlying_type<T> error;
    std::cout << "eta = " << eta << std::endl;
    std::cout << "task_based_cholesky_factorization tests...\n";

    // Setup test case
    htool::TestCaseSolve<T, GeneratorTestType> test_case('L', 'N', n1, n2, 1, -1);

    // HMatrix
    HMatrixTreeBuilder<T, htool::underlying_type<T>> hmatrix_tree_builder_A(epsilon, eta, is_complex<T>() ? 'H' : 'S', UPLO);
    HMatrix<T, htool::underlying_type<T>> HA = hmatrix_tree_builder_A.build(*test_case.operator_in_user_numbering_A, *test_case.root_cluster_A_output, *test_case.root_cluster_A_input);

    // Matrix
    int ni_A = test_case.root_cluster_A_input->get_size();
    int no_A = test_case.root_cluster_A_output->get_size();
    int ni_X = test_case.root_cluster_X_input->get_size();
    int no_X = test_case.root_cluster_X_output->get_size();
    Matrix<T> A_dense(no_A, ni_A), X_dense(no_X, ni_X), B_dense(X_dense), densified_hmatrix_test(B_dense), matrix_test;
    std::vector<int> identity(ni_A);
    std::iota(identity.begin(), identity.end(), test_case.root_cluster_A_output->get_offset());
    test_case.operator_in_user_numbering_A->copy_submatrix(no_A, ni_A, identity.data(), identity.data(), A_dense.data());
    generate_random_matrix(X_dense);
    add_symmetric_matrix_matrix_product('L', UPLO, T(1.), A_dense, X_dense, T(0.), B_dense);

    // Tests
    std::chrono::steady_clock::time_point start, end;
    auto task_based_HA = HA;

    //// Classic Cholesky factorization
    matrix_test = B_dense;
    start       = std::chrono::steady_clock::now();
    cholesky_factorization(UPLO, HA);
    end = std::chrono::steady_clock::now();
    cholesky_solve(UPLO, HA, matrix_test);

    std::chrono::duration<double> classic_duration = end - start;
    error                                          = normFrob(X_dense - matrix_test) / normFrob(X_dense);
    is_error                                       = is_error || !(error < epsilon);
    cout << ">   classic error = " << error << endl;
    cout << "    classic_duration = " << classic_duration.count() << std::endl;

    //// Task-based Cholesky factorization
    int max_nb_nodes               = 64;
    std::vector<HMatrix<T> *> L0_A = find_l0(task_based_HA, max_nb_nodes);
    matrix_test                    = B_dense;
    start                          = std::chrono::steady_clock::now();
#if defined(_OPENMP) && !defined(HTOOL_WITH_PYTHON_INTERFACE)
#    pragma omp parallel
#    pragma omp single
#endif
    {
        task_based_cholesky_factorization(UPLO, task_based_HA, L0_A);
    }
    end = std::chrono::steady_clock::now();
    cholesky_solve(UPLO, task_based_HA, matrix_test);

    std::chrono::duration<double> task_based_duration = end - start;
    error                                             = normFrob(X_dense - matrix_test) / normFrob(X_dense);
    is_error                                          = is_error || !(error < epsilon);
    cout << ">   task_based error = " << error << endl;
    cout << "    task_based_duration = " << task_based_duration.count() << std::endl;
    if (task_based_duration.count() > classic_duration.count()) {
        std::cerr << "Careful: task_based_duration > classic_duration. Ratio TB/Classic = " << task_based_duration.count() / classic_duration.count() << std::endl;
    }

    // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // Print the results
    // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if (is_error) {
        std::cerr << "ERROR: test_task_based_cholesky_factorization current case failed." << std::endl;
    } else {
        std::cout << "SUCCESS: test_task_based_cholesky_factorization current case passed." << std::endl;
        std::cout << "===============================================================\n"
                  << std::endl;
    }
    return is_error;
} // end of test_task_based_cholesky_factorization

template <typename T, typename GeneratorTestType, std::enable_if_t<is_complex_t<T>::value, bool> = true>
bool test_task_based_cholesky_factorization(char UPLO, int n1, int n2, htool::underlying_type<T> epsilon) {
    bool is_error = false;
    double eta    = 100;
    htool::underlying_type<T> error;
    std::cout << "eta = " << eta << std::endl;
    std::cout << "task_based_cholesky_factorization tests...\n";

    // Setup test case
    htool::TestCaseSolve<T, GeneratorTestType> test_case('L', 'N', n1, n2, 1, -1);

    // HMatrix
    HMatrixTreeBuilder<T, htool::underlying_type<T>> hmatrix_tree_builder_A(epsilon, eta, is_complex<T>() ? 'H' : 'S', UPLO);
    HMatrix<T, htool::underlying_type<T>> HA = hmatrix_tree_builder_A.build(*test_case.operator_in_user_numbering_A, *test_case.root_cluster_A_output, *test_case.root_cluster_A_input);

    // Matrix
    int ni_A = test_case.root_cluster_A_input->get_size();
    int no_A = test_case.root_cluster_A_output->get_size();
    int ni_X = test_case.root_cluster_X_input->get_size();
    int no_X = test_case.root_cluster_X_output->get_size();
    Matrix<T> A_dense(no_A, ni_A), X_dense(no_X, ni_X), B_dense(X_dense), densified_hmatrix_test(B_dense), matrix_test;
    std::vector<int> identity(ni_A);
    std::iota(identity.begin(), identity.end(), test_case.root_cluster_A_output->get_offset());
    test_case.operator_in_user_numbering_A->copy_submatrix(no_A, ni_A, identity.data(), identity.data(), A_dense.data());
    generate_random_matrix(X_dense);
    add_hermitian_matrix_matrix_product('L', UPLO, T(1.), A_dense, X_dense, T(0.), B_dense);

    // Tests
    std::chrono::steady_clock::time_point start, end;
    auto task_based_HA = HA;

    //// Classic Cholesky factorization
    matrix_test = B_dense;
    start       = std::chrono::steady_clock::now();
    cholesky_factorization(UPLO, HA);
    end = std::chrono::steady_clock::now();
    cholesky_solve(UPLO, HA, matrix_test);
    std::chrono::duration<double> classic_duration = end - start;

    error    = normFrob(X_dense - matrix_test) / normFrob(X_dense);
    is_error = is_error || !(error < epsilon);
    cout << ">   classic error = " << error << endl;
    cout << "    classic_duration = " << classic_duration.count() << std::endl;

    //// Task-based Cholesky factorization
    int max_nb_nodes               = 64;
    std::vector<HMatrix<T> *> L0_A = find_l0(task_based_HA, max_nb_nodes);
    matrix_test                    = B_dense;
    start                          = std::chrono::steady_clock::now();
#if defined(_OPENMP) && !defined(HTOOL_WITH_PYTHON_INTERFACE)
#    pragma omp parallel
#    pragma omp single
#endif
    {
        task_based_cholesky_factorization(UPLO, task_based_HA, L0_A);
    }
    end = std::chrono::steady_clock::now();
    cholesky_solve(UPLO, task_based_HA, matrix_test);

    std::chrono::duration<double> task_based_duration = end - start;
    error                                             = normFrob(X_dense - matrix_test) / normFrob(X_dense);
    is_error                                          = is_error || !(error < epsilon);
    cout << ">   task_based error = " << error << endl;
    cout << "    task_based_duration = " << task_based_duration.count() << std::endl;
    if (task_based_duration.count() > classic_duration.count()) {
        std::cerr << "Careful: task_based_duration > classic_duration. Ratio TB/Classic = " << task_based_duration.count() / classic_duration.count() << std::endl;
    }

    // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // Print the results
    // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if (is_error) {
        std::cerr << "ERROR: test_task_based_cholesky_factorization current case failed." << std::endl;
    } else {
        std::cout << "SUCCESS: test_task_based_cholesky_factorization current case passed." << std::endl;
        std::cout << "===============================================================\n"
                  << std::endl;
    }
    return is_error;
}
