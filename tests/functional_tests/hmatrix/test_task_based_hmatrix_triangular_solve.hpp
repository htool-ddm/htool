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
        htool::Logger::get_instance().log(LogLevel::WARNING, "Careful: task_based_duration > classic_duration. Ratio TB/Classic = " + std::to_string(task_based_duration.count() / classic_duration.count()) + "."); // LCOV_EXCL_LINE
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
        htool::Logger::get_instance().log(LogLevel::WARNING, "Careful: task_based_duration > classic_duration. Ratio TB/Classic = " + std::to_string(task_based_duration.count() / classic_duration.count()) + "."); // LCOV_EXCL_LINE
    }

    // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // Print the results
    // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if (is_error) {
        htool::Logger::get_instance().log(LogLevel::ERROR, "test_task_based_hmatrix_triangular_solve current case failed."); // LCOV_EXCL_LINE
    } else {
        std::cout << "SUCCESS: test_task_based_hmatrix_triangular_solve current case passed." << std::endl;
        std::cout << "===============================================================\n"
                  << std::endl;
    }
    return is_error;
} // end of test_task_based_hmatrix_triangular_solve
