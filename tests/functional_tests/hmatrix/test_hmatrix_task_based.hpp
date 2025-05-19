#include <cmath>                             // for pow, sqrt
#include <cstddef>                           // for size_t
#include <htool/basic_types/vector.hpp>      // for norm2
#include <htool/clustering/cluster_node.hpp> // for Cluster...
#include <htool/clustering/cluster_output.hpp>
#include <htool/clustering/tree_builder/recursive_build.hpp> // for Cluster...
#include <htool/hmatrix/hmatrix.hpp>                         // for copy_di...
#include <htool/hmatrix/hmatrix_distributed_output.hpp>      // for print_d...
#include <htool/hmatrix/hmatrix_output.hpp>                  // for print_h...
#include <htool/hmatrix/hmatrix_output_dot.hpp>              // for view_block_tree...
#include <htool/hmatrix/interfaces/virtual_generator.hpp>    // for Generat...
#include <htool/hmatrix/linalg/add_hmatrix_vector_product.hpp>
#include <htool/hmatrix/linalg/task_based_add_hmatrix_vector_product.hpp> // for task_bas...
#include <htool/hmatrix/lrmat/SVD.hpp>
#include <htool/hmatrix/tree_builder/task_based_tree_builder.hpp> // for enumerate_dependence, find_l0...
#include <htool/hmatrix/tree_builder/tree_builder.hpp>            // for HMatrix...
#include <htool/matrix/matrix.hpp>                                // for Matrix
#include <htool/misc/misc.hpp>                                    // for underly...
#include <htool/misc/user.hpp>                                    // for NbrToStr
#include <htool/testing/dense_blocks_generator_test.hpp>
#include <htool/testing/generate_test_case.hpp> // for TestCaseSymmetricPro...
#include <htool/testing/geometry.hpp>           // for create_...
#include <htool/testing/partition.hpp>          // for test_pa...
#include <iostream>                             // for operator<<
#include <memory>                               // for make_sh...
#include <string>                               // for operator+
#include <vector>                               // for vector

using namespace std;
using namespace htool;

template <typename T, typename GeneratorTestType>
bool test_hmatrix_task_based(char transa, int n1, int n2, htool::underlying_type<T> epsilon, bool block_tree_consistency) {
    bool is_error = false;
    int rankWorld = 0;
    double eta    = 10;

    TestCaseProduct<T, GeneratorTestType> test_case(transa, 'N', n1, n2, 1, 1, 2);
    const Cluster<htool::underlying_type<T>> *root_cluster_A_output, *root_cluster_A_input;

    root_cluster_A_output = test_case.root_cluster_A_output;
    root_cluster_A_input  = test_case.root_cluster_A_input;
    int ni_A              = test_case.ni_A;
    int no_A              = test_case.no_A;
    // int ni_B              = test_case.ni_B;
    int no_B = test_case.no_B;
    // int ni_C              = test_case.ni_C;
    int no_C = test_case.no_C;

    // Tree builder
    HMatrixTreeBuilder<T, htool::underlying_type<T>> hmatrix_tree_builder(*root_cluster_A_output, *root_cluster_A_input, epsilon, eta, 'N', 'N', -1, -1, rankWorld);
    hmatrix_tree_builder.set_low_rank_generator(std::make_shared<SVD<T>>());
    auto *input_cluster  = &hmatrix_tree_builder.get_source_cluster();
    auto *output_cluster = &hmatrix_tree_builder.get_target_cluster();
    if (transa != 'N') {
        std::swap(input_cluster, output_cluster);
    }
    hmatrix_tree_builder.set_block_tree_consistency(block_tree_consistency);

    // build
    HMatrix<T, htool::underlying_type<T>> root_hmatrix = hmatrix_tree_builder.build(*test_case.operator_A);

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

    // Tests for task_based_compute_blocks
    {
        std::cout << "task_based_compute_blocks tests...";
        // build
        HMatrix<T> task_based_hmatrix(*root_cluster_A_output, *root_cluster_A_input);

#if defined(_OPENMP) && !defined(HTOOL_WITH_PYTHON_INTERFACE)
#    pragma omp parallel
#    pragma omp single
#endif
        {
            task_based_hmatrix = hmatrix_tree_builder.build(*test_case.operator_A, true);
        }
        task_based_hmatrix.get_hmatrix_tree_data()->m_information["Number_of_false_positive"] = NbrToStr(hmatrix_tree_builder.get_false_positive()); // when task_based_compute_blocks is used in build, the number of false positives is not set in the hmatrix tree data because the tasks need to finish before we can know the number of false positive

        auto hmatrix = hmatrix_tree_builder.build(*test_case.operator_A); // when compute_blocks is used in build, the number of false positives is set in the hmatrix tree data

        // densification
        Matrix<T> densified_hmatrix(no_A, ni_A), densified_hmatrix_task_based(no_A, ni_A);
        copy_to_dense(hmatrix, densified_hmatrix.data());
        copy_to_dense(task_based_hmatrix, densified_hmatrix_task_based.data());

        // compare
        is_error = is_error || (normFrob(densified_hmatrix - densified_hmatrix_task_based) > 1e-10);
        // is_error = is_error || (hmatrix.get_hmatrix_tree_data()->m_information["Number_of_false_positive"] != task_based_hmatrix.get_hmatrix_tree_data()->m_information["Number_of_false_positive"]);

        if (is_error) {
            std::cout << "ERROR" << std::endl;
        } else {
            std::cout << "SUCCESS" << std::endl;
        }
        std::cout << "    hmatrix m_false_positive = " << hmatrix.get_hmatrix_tree_data()->m_information["Number_of_false_positive"] << std::endl;
        std::cout << "    task_based_hmatrix m_false_positive = " << task_based_hmatrix.get_hmatrix_tree_data()->m_information["Number_of_false_positive"] << std::endl;

        // check durations
        std::chrono::duration<double> classic_duration    = hmatrix.get_hmatrix_tree_data()->m_timings["Blocks_computation_walltime"];
        std::chrono::duration<double> task_based_duration = task_based_hmatrix.get_hmatrix_tree_data()->m_timings["Blocks_computation_walltime"];
        if (task_based_duration.count() > classic_duration.count()) {
            std::cerr << "Careful: task_based_duration: " << task_based_duration.count() << " > classic_duration: " << classic_duration.count() << "." << std::endl;
        }

        std::cout << "----------------------------------" << std::endl;
    }

    // Tests for task_based_internal_add_hmatrix_vector_product
    {
        std::cout << "task_based_internal_add_hmatrix_vector_product tests...";

        // build
        auto root_hmatrix_classic    = hmatrix_tree_builder.build(*test_case.operator_A);
        auto root_hmatrix_task_based = hmatrix_tree_builder.build(*test_case.operator_A, true);
        // print_hmatrix_information(root_hmatrix_task_based, std::cout);

        // Create a vector for the input and output
        std::vector<T> in(no_B), out(no_C, 11), out_task(no_C, 11);
        for (int i = 0; i < no_B; i++) {
            in[i] = i * i;
        }

        // Create a vector for the expected output
        T alpha = T(3);
        T beta  = T(2);
        openmp_internal_add_hmatrix_vector_product(transa, alpha, root_hmatrix_classic, in.data(), beta, out.data());

        // L0 definitions
        std::vector<HMatrix<T> *> L0           = hmatrix_tree_builder.get_L0();
        std::vector<const Cluster<T> *> in_L0  = find_l0(*input_cluster, 64);
        std::vector<const Cluster<T> *> out_L0 = find_l0(*output_cluster, 64);

        std::mutex cout_mutex; // mutex to protect cout

#if defined(_OPENMP) && !defined(HTOOL_WITH_PYTHON_INTERFACE)
#    pragma omp parallel
#    pragma omp single
#endif
        {
            // Perform the task-based internal add H-matrix vector product
            task_based_internal_add_hmatrix_vector_product(transa, alpha, root_hmatrix_task_based, in.data(), beta, out_task.data(), L0, in_L0, out_L0);
        }

        // Compare the results
        is_error = is_error || (norm2(out - out_task) / norm2(out) > 1e-15);
        if (is_error) {
            std::cout << "ERROR" << std::endl;
        } else {
            std::cout << "SUCCESS" << std::endl;
        }
        std::cout << "    norm2(out ) = " << norm2(out) << std::endl;
        std::cout << "    norm2(out - out_task)/norm2(out) = " << norm2(out - out_task) / norm2(out) << std::endl;
        std::cout << "----------------------------------" << std::endl;
    }

    // Tests for hmatrix vector product following assembly with task_based_compute_blocks and task_based_internal_add_hmatrix_vector_product
    {
        std::cout << "assembly + hmatrix vector products tests...";
        std::chrono::duration<double> disjoint_case_duration, conjoint_case_duration;
        // Case 1 : task graphs are disjoint
        {
            T alpha = T(3);
            T beta  = T(2);
            std::vector<T> in(no_B), out(no_C, 11), out_task(no_C, 11);
            for (int i = 0; i < no_B; i++) {
                in[i] = i * i;
            }
            std::mutex cout_mutex; // mutex to protect cout

            // build
            HMatrix<T> task_based_hmatrix(*root_cluster_A_output, *root_cluster_A_input);

            std::chrono::steady_clock::time_point start, end;
            start = std::chrono::steady_clock::now();
#if defined(_OPENMP) && !defined(HTOOL_WITH_PYTHON_INTERFACE)
#    pragma omp parallel
#    pragma omp single
#endif
            {
                task_based_hmatrix = hmatrix_tree_builder.build(*test_case.operator_A, true);
            }

            // N hmatrix vector products
            std::vector<HMatrix<T> *> L0           = hmatrix_tree_builder.get_L0();
            std::vector<const Cluster<T> *> in_L0  = find_l0(*input_cluster, 64);
            std::vector<const Cluster<T> *> out_L0 = find_l0(*output_cluster, 64);

#if defined(_OPENMP) && !defined(HTOOL_WITH_PYTHON_INTERFACE)
#    pragma omp parallel
#    pragma omp single
#endif
            {
                for (int i = 0; i < 100; i++) {
                    task_based_internal_add_hmatrix_vector_product(transa, alpha, task_based_hmatrix, in.data(), beta, out_task.data(), L0, in_L0, out_L0);
                }
            }
            end                    = std::chrono::steady_clock::now();
            disjoint_case_duration = end - start;
            std::cout << "\n    disjoint_case_duration = " << disjoint_case_duration.count() << std::endl;
        } // end of case 1

        // Case 2 : task graphs are not conjoint
        {
            T alpha = T(3);
            T beta  = T(2);
            std::vector<T> in(no_B), out(no_C, 11), out_task(no_C, 11);
            for (int i = 0; i < no_B; i++) {
                in[i] = i * i;
            }
            std::mutex cout_mutex; // mutex to protect cout

            // build
            HMatrix<T> task_based_hmatrix(*root_cluster_A_output, *root_cluster_A_input);

            std::chrono::steady_clock::time_point start, end;
            start = std::chrono::steady_clock::now();
#if defined(_OPENMP) && !defined(HTOOL_WITH_PYTHON_INTERFACE)
#    pragma omp parallel
#    pragma omp single
#endif
            {
                task_based_hmatrix = hmatrix_tree_builder.build(*test_case.operator_A, true);

                // N hmatrix vector products
                std::vector<HMatrix<T> *> L0           = hmatrix_tree_builder.get_L0();
                std::vector<const Cluster<T> *> in_L0  = find_l0(*input_cluster, 64);
                std::vector<const Cluster<T> *> out_L0 = find_l0(*output_cluster, 64);

                for (int i = 0; i < 100; i++) {
                    task_based_internal_add_hmatrix_vector_product(transa, alpha, task_based_hmatrix, in.data(), beta, out_task.data(), L0, in_L0, out_L0);
                }

                end                    = std::chrono::steady_clock::now();
                conjoint_case_duration = end - start;
                std::cout << "    conjoint_case_duration = " << conjoint_case_duration.count() << std::endl;
            }
        } // end of case 2

        if (disjoint_case_duration.count() < conjoint_case_duration.count()) {
            std::cerr << "Careful: disjoint_case_duration: " << disjoint_case_duration.count() << " < conjoint_case_duration: " << conjoint_case_duration.count() << "." << std::endl;
        }

        std::cout << "----------------------------------" << std::endl;
    }

    // Print the results
    if (is_error) {
        std::cerr << "ERROR: test_hmatrix_task_based current case failed." << std::endl;
    } else {
        std::cout << "SUCCESS: test_hmatrix_task_based current case passed." << std::endl;
        std::cout << "===============================================================\n"
                  << std::endl;
    }
    return is_error;
}

template <typename T, typename GeneratorTestType>
bool test_symmetric_hmatrix_task_based(char transa, int n1, htool::underlying_type<T> epsilon, char UPLO) {
    bool is_error = false;
    int rankWorld = 0;
    double eta    = 10;

    TestCaseSymmetricProduct<T, GeneratorTestType> test_case(n1, 1, 2, 'L', 'S', UPLO);
    const Cluster<htool::underlying_type<T>> *root_cluster_A_output, *root_cluster_A_input;

    root_cluster_A_output = test_case.root_cluster_A_output;
    root_cluster_A_input  = test_case.root_cluster_A_input;
    int ni_A              = test_case.ni_A;
    int no_A              = test_case.no_A;
    // int ni_B              = test_case.ni_B;
    int no_B = test_case.no_B;
    // int ni_C              = test_case.ni_C;
    int no_C = test_case.no_C;

    // Tree builder
    HMatrixTreeBuilder<T, htool::underlying_type<T>> hmatrix_tree_builder(*root_cluster_A_output, *root_cluster_A_input, epsilon, eta, 'S', UPLO, -1, -1, rankWorld);
    hmatrix_tree_builder.set_low_rank_generator(std::make_shared<SVD<T>>());
    auto *input_cluster  = &hmatrix_tree_builder.get_source_cluster();
    auto *output_cluster = &hmatrix_tree_builder.get_target_cluster();
    if (transa != 'N') {
        std::swap(input_cluster, output_cluster);
    }

    // Tests for task_based_compute_blocks
    {
        std::cout << "task_based_compute_blocks tests...";
        // build
        HMatrix<T> task_based_hmatrix(*root_cluster_A_output, *root_cluster_A_input);

#if defined(_OPENMP) && !defined(HTOOL_WITH_PYTHON_INTERFACE)
#    pragma omp parallel
#    pragma omp single
#endif
        {
            task_based_hmatrix = hmatrix_tree_builder.build(*test_case.operator_A, true);
        }
        task_based_hmatrix.get_hmatrix_tree_data()->m_information["Number_of_false_positive"] = NbrToStr(hmatrix_tree_builder.get_false_positive()); // when task_based_compute_blocks is used in build, the number of false positives is not set in the hmatrix tree data because the tasks need to finish before we can know the number of false positive

        auto hmatrix = hmatrix_tree_builder.build(*test_case.operator_A);

        // densification
        Matrix<T> densified_hmatrix(no_A, ni_A), densified_hmatrix_task_based(no_A, ni_A);
        copy_to_dense(hmatrix, densified_hmatrix.data());
        copy_to_dense(task_based_hmatrix, densified_hmatrix_task_based.data());

        // visualization
        std::ofstream build_file("test_build");
        densified_hmatrix_task_based.print(build_file, ",");

        // compare
        is_error = is_error || (normFrob(densified_hmatrix - densified_hmatrix_task_based) > 1e-10);
        // is_error = is_error || (hmatrix.get_hmatrix_tree_data()->m_information["Number_of_false_positive"] != task_based_hmatrix.get_hmatrix_tree_data()->m_information["Number_of_false_positive"]);

        if (is_error) {
            std::cout << "ERROR" << std::endl;
        } else {
            std::cout << "SUCCESS" << std::endl;
        }
        std::cout << "    hmatrix m_false_positive = " << hmatrix.get_hmatrix_tree_data()->m_information["Number_of_false_positive"] << std::endl;
        std::cout << "    task_based_hmatrix m_false_positive = " << task_based_hmatrix.get_hmatrix_tree_data()->m_information["Number_of_false_positive"] << std::endl;

        // check durations
        std::chrono::duration<double> classic_duration    = hmatrix.get_hmatrix_tree_data()->m_timings["Blocks_computation_walltime"];
        std::chrono::duration<double> task_based_duration = task_based_hmatrix.get_hmatrix_tree_data()->m_timings["Blocks_computation_walltime"];
        if (task_based_duration.count() > classic_duration.count()) {
            std::cerr << "Careful: task_based_duration: " << task_based_duration.count() << " > classic_duration: " << classic_duration.count() << "." << std::endl;
        }

        std::cout << "----------------------------------" << std::endl;
    }

    // Tests for task_based_internal_add_hmatrix_vector_product
    {
        std::cout << "task_based_internal_add_hmatrix_vector_product tests...";

        // build
        auto root_hmatrix_classic    = hmatrix_tree_builder.build(*test_case.operator_A);
        auto root_hmatrix_task_based = hmatrix_tree_builder.build(*test_case.operator_A, true);
        // print_hmatrix_information(root_hmatrix_task_based, std::cout);

        // Create a vector for the input and output
        std::vector<T> in(no_B), out(no_C, 11), out_task(no_C, 11);
        for (int i = 0; i < no_B; i++) {
            in[i] = i * i;
        }

        // Create a vector for the expected output
        T alpha = T(3);
        T beta  = T(2);
        openmp_internal_add_hmatrix_vector_product(transa, alpha, root_hmatrix_classic, in.data(), beta, out.data());

        // L0 definitions
        std::vector<HMatrix<T> *> L0           = hmatrix_tree_builder.get_L0();
        std::vector<const Cluster<T> *> in_L0  = find_l0(*input_cluster, 64);
        std::vector<const Cluster<T> *> out_L0 = find_l0(*output_cluster, 64);

        std::mutex cout_mutex; // mutex to protect cout

#if defined(_OPENMP) && !defined(HTOOL_WITH_PYTHON_INTERFACE)
#    pragma omp parallel
#    pragma omp single
#endif
        {
            // Perform the task-based internal add H-matrix vector product
            task_based_internal_add_hmatrix_vector_product(transa, alpha, root_hmatrix_task_based, in.data(), beta, out_task.data(), L0, in_L0, out_L0);
        }

        // Compare the results
        is_error = is_error || (norm2(out - out_task) / norm2(out) > 1e-15);
        if (is_error) {
            std::cout << "ERROR" << std::endl;
        } else {
            std::cout << "SUCCESS" << std::endl;
        }
        std::cout << "    norm2(out ) = " << norm2(out) << std::endl;
        std::cout << "    norm2(out - out_task)/norm2(out) = " << norm2(out - out_task) / norm2(out) << std::endl;
        std::cout << "----------------------------------" << std::endl;
    }

    // Print the results
    if (is_error) {
        std::cerr << "ERROR: test_symmetric_hmatrix_task_based current case failed." << std::endl;
    } else {
        std::cout << "SUCCESS: test_symmetric_hmatrix_task_based current case passed." << std::endl;
        std::cout << "===============================================================\n"
                  << std::endl;
    }
    return is_error;
}
