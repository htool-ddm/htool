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
#include <htool/hmatrix/tree_builder/task_based_tree_builder.hpp>         // for enumerate_dependence, find_l0...
#include <htool/hmatrix/tree_builder/tree_builder.hpp>                    // for HMatrix...
#include <htool/matrix/matrix.hpp>                                        // for Matrix
#include <htool/misc/misc.hpp>                                            // for underly...
#include <htool/misc/user.hpp>                                            // for NbrToStr
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
    // cluster_tree_builder.set_minclustersize(2);

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
    std::cout << "OK" << "\n----------------------------------" << std::endl;

    // Tests for left_hmatrix_descendant_of_right_hmatrix
    std::cout << "left_hmatrix_descendant_of_right_hmatrix tests...";
    is_error = is_error || !(left_hmatrix_descendant_of_right_hmatrix(child1, root_hmatrix));
    is_error = is_error || !(left_hmatrix_descendant_of_right_hmatrix(child2, root_hmatrix));
    is_error = is_error || !(left_hmatrix_descendant_of_right_hmatrix(child1_child1, root_hmatrix));
    is_error = is_error || !(left_hmatrix_descendant_of_right_hmatrix(child1_child2, root_hmatrix));
    std::cout << "OK" << "\n----------------------------------" << std::endl;

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

        std::cout << "OK" << "\n----------------------------------" << std::endl;
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
        HMatrix<T> task_based_hmatrix(*target_root_cluster, *source_root_cluster);

#if defined(_OPENMP) && !defined(HTOOL_WITH_PYTHON_INTERFACE)
#    pragma omp parallel
#    pragma omp single
#endif
        {
            task_based_hmatrix = hmatrix_tree_builder.build(generator, true);
        }
        auto hmatrix = hmatrix_tree_builder.build(generator);

        // densification
        Matrix<T> densified_hmatrix(nr, nc), densified_hmatrix_task_based(nr, nc);
        copy_to_dense(hmatrix, densified_hmatrix.data());
        copy_to_dense(task_based_hmatrix, densified_hmatrix_task_based.data());

        // compare
        is_error = is_error || (normFrob(densified_hmatrix - densified_hmatrix_task_based) > 1e-10);
        // is_error = is_error || (hmatrix.get_hmatrix_tree_data()->m_information["Number_of_false_positive"] != task_based_hmatrix.get_hmatrix_tree_data()->m_information["Number_of_false_positive"]);

        std::cout << "OK" << std::endl;
        std::cout << "    hmatrix m_false_positive = " << hmatrix.get_hmatrix_tree_data()->m_information["Number_of_false_positive"] << std::endl;
        std::cout << "    task_based_hmatrix m_false_positive = " << task_based_hmatrix.get_hmatrix_tree_data()->m_information["Number_of_false_positive"] << std::endl;

        // check durations
        std::chrono::duration<double> classic_duration    = hmatrix.get_hmatrix_tree_data()->m_timings["Blocks_computation_walltime"];
        std::chrono::duration<double> task_based_duration = task_based_hmatrix.get_hmatrix_tree_data()->m_timings["Blocks_computation_walltime"];
        if (task_based_duration > classic_duration) {
            std::cerr << "Warning: task_based_duration: " << task_based_duration.count() << " > classic_duration: " << classic_duration.count() << "." << std::endl;
        }

        std::cout << "----------------------------------" << std::endl;
    }

    // Tests for task_based_internal_add_hmatrix_vector_product
    {
        std::cout << "task_based_internal_add_hmatrix_vector_product tests...";
        char trans = 'T';

        // build
        auto root_hmatrix_classic    = hmatrix_tree_builder.build(generator);
        auto root_hmatrix_task_based = hmatrix_tree_builder.build(generator);
        // print_hmatrix_information(root_hmatrix_task_based, std::cout);

        // Create a vector for the input and output
        std::vector<T> in(nc), out(nr, 11), out_task(nr, 11);
        for (int i = 0; i < nc; i++) {
            in[i] = i * i;
        }

        // Create a vector for the expected output
        openmp_internal_add_hmatrix_vector_product(trans, T(3), root_hmatrix_classic, in.data(), T(0), out.data());

        // L0 definitions
        std::vector<HMatrix<T> *> L0           = find_l0(root_hmatrix_classic, 256);
        std::vector<const Cluster<T> *> in_L0  = find_l0(hmatrix_tree_builder.get_source_cluster(), 64);
        std::vector<const Cluster<T> *> out_L0 = find_l0(hmatrix_tree_builder.get_target_cluster(), 64);
        std::mutex cout_mutex; // mutex to protect cout

#if defined(_OPENMP) && !defined(HTOOL_WITH_PYTHON_INTERFACE)
#    pragma omp parallel
#    pragma omp single
#endif
        {
            // Perform the task-based internal add H-matrix vector product
            task_based_internal_add_hmatrix_vector_product(trans, T(3), root_hmatrix_task_based, in.data(), T(0), out_task.data(), L0, in_L0, out_L0);
        }

        // Compare the results
        is_error = is_error || (norm2(out - out_task) / norm2(out) > 1e-15);
        std::cout << "OK" << std::endl;
        std::cout << "    norm2(out ) = " << norm2(out) << std::endl;
        std::cout << "    norm2(out - out_task)/norm2(out) = " << norm2(out - out_task) / norm2(out) << std::endl;
        std::cout << "----------------------------------" << std::endl;
    }

    // Tests for hmatrix vector product following assembly with task_based_compute_blocks and task_based_internal_add_hmatrix_vector_product
    {
        std::cout << "assembly + hmatrix vector products tests...";
        // Case 1 : task graphs are disjoint
        //         {
        //             // build
        //             HMatrix<T> task_based_hmatrix(*target_root_cluster, *source_root_cluster);

        //             std::chrono::steady_clock::time_point start, end;
        //             start = std::chrono::steady_clock::now();
        // #if defined(_OPENMP) && !defined(HTOOL_WITH_PYTHON_INTERFACE)
        // #    pragma omp parallel
        // #    pragma omp single
        // #endif
        //             {
        //                 task_based_hmatrix = hmatrix_tree_builder.build(generator, true);
        //             }

        //             // N hmatrix vector products
        //             std::vector<T> in(nc), out(nr, 11), out_task(nr, 11);
        //             for (int i = 0; i < nc; i++) {
        //                 in[i] = i * i;
        //             }
        //             std::vector<HMatrix<T> *> L0           = find_l0(root_hmatrix, 256); // Todo : Change to get back L0 from task_based_hmatrix
        //             std::vector<const Cluster<T> *> in_L0  = find_l0(hmatrix_tree_builder.get_source_cluster(), 64);
        //             std::vector<const Cluster<T> *> out_L0 = find_l0(hmatrix_tree_builder.get_target_cluster(), 64);
        //             std::mutex cout_mutex; // mutex to protect cout

        // #if defined(_OPENMP) && !defined(HTOOL_WITH_PYTHON_INTERFACE)
        // #    pragma omp parallel
        // #    pragma omp single
        // #endif
        //             {
        //                 for (int i = 0; i < 100; i++) {
        //                     task_based_internal_add_hmatrix_vector_product('N', T(3), task_based_hmatrix, in.data(), T(3), out_task.data(), L0, in_L0, out_L0);
        //                 }
        //             }
        //             end                                                  = std::chrono::steady_clock::now();
        //             std::chrono::duration<double> disjoint_case_duration = end - start;
        //             std::cout << "    disjoint_case_duration = " << disjoint_case_duration.count() << std::endl;
        //             // #pragma omp taskwait
        //         }

        std::cout << "OK" << "\n----------------------------------" << std::endl;
    }

    // Print the results
    if (is_error) {
        std::cerr << "ERROR: test_hmatrix_task_based failed." << std::endl;
    } else {
        std::cout << "SUCCESS: test_hmatrix_task_based passed." << "\n================================" << std::endl;
    }
    return is_error;
}
