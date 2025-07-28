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
            htool::Logger::get_instance().log(LogLevel::ERROR, "Careful: task_based_duration > classic_duration. Ratio TB/Classic = " + std::to_string(task_based_duration.count() / classic_duration.count()) + "."); // LCOV_EXCL_LINE
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
            htool::Logger::get_instance().log(LogLevel::ERROR, "Careful: task_based_duration > classic_duration. Ratio TB/Classic = " + std::to_string(task_based_duration.count() / classic_duration.count()) + "."); // LCOV_EXCL_LINE
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
            htool::Logger::get_instance().log(LogLevel::ERROR, "Careful: disjoint_case_TB_duration < conjoint_case_TB_duration. Ratio TB/Classic = " + std::to_string(conjoint_case_duration.count() / disjoint_case_duration.count()) + "."); // LCOV_EXCL_LINE
        }

        std::cout << "----------------------------------" << std::endl;
    }

    // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // Print the results
    // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if (is_error) {
        htool::Logger::get_instance().log(LogLevel::ERROR, " test_task_based_hmatrix_vector_product current case failed."); // LCOV_EXCL_LINE
    } else {
        std::cout << "SUCCESS: test_task_based_hmatrix_vector_product current case passed." << std::endl;
        std::cout << "===============================================================\n"
                  << std::endl;
    }
    return is_error;
}
