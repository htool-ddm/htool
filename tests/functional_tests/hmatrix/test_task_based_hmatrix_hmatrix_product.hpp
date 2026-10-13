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
// #include <htool/hmatrix/linalg/factorization.hpp>
#include <htool/hmatrix/linalg/task_based_add_hmatrix_hmatrix_product.hpp> // for task_bas...
#include <htool/hmatrix/linalg/task_based_add_hmatrix_vector_product.hpp>  // for task_bas...
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
bool test_task_based_hmatrix_hmatrix_product(const TestCaseType &test_case, char sym, char transa, char transb, htool::underlying_type<T> epsilon, bool block_tree_consistency, char UPLO = 'L') {
    bool is_error = false;
    double eta    = 10;
    std::cout << "eta = " << eta << '\n';
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
    auto hmatrix_A            = hmatrix_tree_builder_A->build(*test_case.operator_A, *root_cluster_A_output, *root_cluster_A_input);
    auto hmatrix_B            = hmatrix_tree_builder_B->build(*test_case.operator_B, *root_cluster_B_output, *root_cluster_B_input);
    auto hmatrix_classic_C    = hmatrix_tree_builder_C->build(*test_case.operator_C, *root_cluster_C_output, *root_cluster_C_input);
    auto hmatrix_task_based_C = hmatrix_classic_C;

    std::vector<const HMatrix<T> *> L0_A = find_l0(hmatrix_A, 64);
    std::vector<const HMatrix<T> *> L0_B = find_l0(hmatrix_B, 64);
    std::vector<const HMatrix<T> *> L0_C = find_l0(hmatrix_task_based_C, 64);
    // parameters
    T alpha         = T(3);
    T beta          = T(2);
    int nb_products = 1;

    std::chrono::steady_clock::time_point start, end;

    // Perform the classic hmatrix hmatrix product
    start = std::chrono::steady_clock::now();
    for (int i = 0; i < nb_products; i++) {
        internal_add_hmatrix_hmatrix_product(transa, transb, alpha, hmatrix_A, hmatrix_B, beta, hmatrix_classic_C);
    }
    end                                            = std::chrono::steady_clock::now();
    std::chrono::duration<double> classic_duration = end - start;

    // Perform the task-based internal add H-matrix hamtrix product
    start = std::chrono::steady_clock::now();
#if defined(_OPENMP)
#    pragma omp parallel
#    pragma omp single
#endif
    {
        for (int i = 0; i < nb_products; i++) {
            task_based_internal_add_hmatrix_hmatrix_product(transa, transb, alpha, hmatrix_A, hmatrix_B, beta, hmatrix_task_based_C, L0_A, L0_B, L0_C);
        }
    }
    end                                               = std::chrono::steady_clock::now();
    std::chrono::duration<double> task_based_duration = end - start;

    // desify results
    Matrix<T> densified_hmatrix_classic_C(no_C, ni_C), densified_hmatrix_task_based_C(no_C, ni_C);
    copy_to_dense(hmatrix_classic_C, densified_hmatrix_classic_C.data());
    copy_to_dense(hmatrix_task_based_C, densified_hmatrix_task_based_C.data());

    // Compare the results
    htool::underlying_type<T> error = normFrob(densified_hmatrix_classic_C - densified_hmatrix_task_based_C) / normFrob(densified_hmatrix_classic_C);
    std::cout << "(error: " << error << ") ";
    is_error = is_error || !(error < 1e-12);

    // Print the results
    if (is_error) {
        std::cout << "ERROR" << '\n';
    } else {
        std::cout << "SUCCESS" << '\n';
    }

    // beta != 1 on a sub-block of C, while L0_C is the L0 of the whole C: only this block is scaled
    if (sym == 'N' && transa == 'N' && transb == 'N' && !root_cluster_C_output->is_leaf() && !root_cluster_A_input->is_leaf() && !root_cluster_C_input->is_leaf()) {
        const Cluster<htool::underlying_type<T>> &t = *root_cluster_C_output->get_children()[0];
        const Cluster<htool::underlying_type<T>> &m = *root_cluster_A_input->get_children()[0];
        const Cluster<htool::underlying_type<T>> &s = *root_cluster_C_input->get_children()[0];
        const HMatrix<T> *A_sub                     = hmatrix_A.get_sub_hmatrix(t, m);
        const HMatrix<T> *B_sub                     = hmatrix_B.get_sub_hmatrix(m, s);
        HMatrix<T> *C_sub_classic                   = hmatrix_classic_C.get_sub_hmatrix(t, s);
        HMatrix<T> *C_sub_task_based                = hmatrix_task_based_C.get_sub_hmatrix(t, s);
        if (A_sub != nullptr && B_sub != nullptr && C_sub_classic != nullptr && C_sub_task_based != nullptr) {
            internal_add_hmatrix_hmatrix_product('N', 'N', alpha, *A_sub, *B_sub, beta, *C_sub_classic);
#if defined(_OPENMP)
#    pragma omp parallel
#    pragma omp single
#endif
            {
                task_based_internal_add_hmatrix_hmatrix_product('N', 'N', alpha, *A_sub, *B_sub, beta, *C_sub_task_based, L0_A, L0_B, L0_C);
            }
            copy_to_dense(hmatrix_classic_C, densified_hmatrix_classic_C.data());
            copy_to_dense(hmatrix_task_based_C, densified_hmatrix_task_based_C.data());
            htool::underlying_type<T> sub_block_error = normFrob(densified_hmatrix_classic_C - densified_hmatrix_task_based_C) / normFrob(densified_hmatrix_classic_C);
            std::cout << "> Error on a product with beta != 1 on a sub-block of C: " << sub_block_error << '\n';
            is_error = is_error || !(sub_block_error < 1e-12);
        }
    }

    std::cout << "    normFrob(densified_hmatrix_classic_C) = " << normFrob(densified_hmatrix_classic_C) << '\n';
    std::cout << "    normFrob(densified_hmatrix_task_based_C) = " << normFrob(densified_hmatrix_task_based_C) << '\n';

    std::cout << "    normFrob(classic_C - task_based_C) / normFrob(classic_C) = " << normFrob(densified_hmatrix_classic_C - densified_hmatrix_task_based_C) / normFrob(densified_hmatrix_classic_C) << '\n'
              << '\n';

    std::cout << "    classic_duration = " << classic_duration.count() << '\n';
    std::cout << "    task_based_duration = " << task_based_duration.count() << '\n';

    // check durations
    if (task_based_duration.count() > classic_duration.count()) {
        htool::Logger::get_instance().log(LogLevel::WARNING, "Careful: task_based_duration > classic_duration. Ratio TB/Classic = " + std::to_string(task_based_duration.count() / classic_duration.count()) + "."); // LCOV_EXCL_LINE
    }
    std::cout << "----------------------------------" << '\n';

    // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // Print the results
    // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if (is_error) {
        htool::Logger::get_instance().log(LogLevel::ERROR, "test_task_based_hmatrix_hmatrix_product current case failed."); // LCOV_EXCL_LINE
    } else {
        std::cout << "SUCCESS: test_task_based_hmatrix_hmatrix_product current case passed." << '\n';
        std::cout << "===============================================================\n"
                  << '\n';
    }
    return is_error;
}
