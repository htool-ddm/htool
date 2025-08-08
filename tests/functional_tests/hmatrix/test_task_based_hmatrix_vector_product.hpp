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
bool test_task_based_hmatrix_vector_product(const TestCaseType &test_case, char sym, char transa, htool::underlying_type<T> epsilon, bool block_tree_consistency, char UPLO = 'L') {
    double eta = 10;
    std::cout << "eta = " << eta << std::endl;
    double error_tol = 1e-14;
    bool is_error    = false;

    // Get test case parameters
    const Cluster<htool::underlying_type<T>> *root_cluster_A_output, *root_cluster_A_input;
    root_cluster_A_output = test_case.root_cluster_A_output;
    root_cluster_A_input  = test_case.root_cluster_A_input;
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

    // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // Tests for task_based_internal_add_hmatrix_vector_product
    // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    std::cout << "task_based_internal_add_hmatrix_vector_product tests...";
    std::chrono::steady_clock::time_point start, end;

    // build
    auto hmatrix                 = hmatrix_tree_builder->build(*test_case.operator_A, *root_cluster_A_output, *root_cluster_A_input);
    std::vector<HMatrix<T> *> L0 = find_l0(hmatrix, 64);

    // L0 definitions
    std::vector<const Cluster<htool::underlying_type<T>> *> in_L0  = find_l0(*input_cluster, 8);
    std::vector<const Cluster<htool::underlying_type<T>> *> out_L0 = find_l0(*output_cluster, 8);

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
        openmp_internal_add_hmatrix_vector_product(transa, alpha, hmatrix, in.data(), beta, out.data());
    }
    end = std::chrono::steady_clock::now();

    std::chrono::duration<double> classic_duration = end - start;

    // Perform the task-based internal add H-matrix vector product
    start = std::chrono::steady_clock::now();
#if defined(_OPENMP) && !defined(HTOOL_WITH_PYTHON_INTERFACE)
#    pragma omp parallel
#    pragma omp single
#endif
    {
        for (int i = 0; i < nb_products; i++) {
            task_based_internal_add_hmatrix_vector_product(transa, alpha, hmatrix, in.data(), beta, out_task.data(), L0, in_L0, out_L0);
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
    // std::cout << "    norm2(out) = " << norm2(out) << std::endl;

    std::cout
        << "    norm2(out - out_task)/norm2(out) = " << norm2(out - out_task) / norm2(out) << std::endl
        << std::endl;

    std::cout << "    classic_duration = " << classic_duration.count() << std::endl;
    std::cout << "    task_based_duration = " << task_based_duration.count() << std::endl;

    // check durations
    if (task_based_duration.count() > classic_duration.count()) {
        htool::Logger::get_instance().log(LogLevel::WARNING, "Careful: task_based_duration > classic_duration. Ratio TB/Classic = " + std::to_string(task_based_duration.count() / classic_duration.count()) + "."); // LCOV_EXCL_LINE
    }
    std::cout << "----------------------------------" << std::endl;

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
