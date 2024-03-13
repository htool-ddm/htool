
#include <htool/clustering/recursive_build_strategies/recursive_build_strategies.hpp>
#include <htool/local_operators/local_dense_matrix.hpp>
#include <htool/testing/generator_test.hpp>
#include <htool/testing/geometry.hpp>
#include <mpi.h>
#include <random>

using namespace std;
using namespace htool;

template <typename CoefficientPrecision>
int test_local_operator_dense_product(int number_of_rows, int number_of_columns, int mu, char op, char symmetry, char UPLO) {

    int sizeWorld;
    MPI_Comm_size(MPI_COMM_WORLD, &sizeWorld);
    srand(1);
    bool test = 0;

    // Geometry
    int dim = 3;
    std::vector<htool::underlying_type<CoefficientPrecision>> xt(dim * number_of_rows);
    std::vector<htool::underlying_type<CoefficientPrecision>> xs((symmetry == 'N' && number_of_columns != number_of_rows) ? dim * number_of_columns : 0);

    htool::underlying_type<CoefficientPrecision> z1 = 1;
    create_disk(dim, z1, number_of_rows, xt.data());

    // Cluster
    ClusterTreeBuilder<double, ComputeLargestExtent<double>, RegularSplitting<double>> recursive_build_strategy(number_of_rows, dim, xt.data(), 2, sizeWorld);

    std::shared_ptr<ClusterTree<double>> target_cluster_tree = make_shared<ClusterTree<double>>(recursive_build_strategy.create_cluster_tree());
    std::shared_ptr<ClusterTree<double>> source_cluster_tree;

    if (symmetry == 'N' && number_of_rows != number_of_columns) {
        htool::underlying_type<CoefficientPrecision> z2 = z1 + 0.1;
        create_disk(dim, z2, number_of_columns, xs.data());
        ClusterTreeBuilder<double, ComputeLargestExtent<double>, RegularSplitting<double>> source_recursive_build_strategy(number_of_columns, dim, xs.data(), 2, sizeWorld);
        source_cluster_tree = make_shared<ClusterTree<double>>(source_recursive_build_strategy.create_cluster_tree());
    } else {
        source_cluster_tree = target_cluster_tree;
    }
    // Generator
    GeneratorTestDoubleSymmetric generator(3, number_of_rows, number_of_columns, xt, xs, target_cluster_tree, source_cluster_tree);

    // LocalDenseMatrix
    LocalDenseMatrix<double> A(generator, target_cluster_tree, source_cluster_tree);

    // Input sizes
    int ni = (op == 'T' || op == 'C') ? number_of_rows : number_of_columns;
    int no = (op == 'T' || op == 'C') ? number_of_columns : number_of_rows;

    // Random input vector
    vector<CoefficientPrecision> x(ni * mu, 1), y(no * mu, 1), ref(no * mu, 1);
    CoefficientPrecision alpha, beta;
    htool::underlying_type<CoefficientPrecision> error;
    generate_random_vector(x);
    generate_random_vector(y);
    generate_random_scalar(alpha);
    generate_random_scalar(beta);
    if (op == 'N') {
        std::vector<CoefficientPrecision> temp(no * mu, 0);
        generator.mvprod(x.data(), temp.data(), mu);
        ref = mult(alpha, temp) + mult(beta, y);
    } else {
        std::vector<CoefficientPrecision> temp(no * mu, 0);
        generator.mvprod_transp(x.data(), temp.data(), mu);
        ref = mult(alpha, temp) + mult(beta, y);
    }

    // Permutation
    vector<CoefficientPrecision> x_perm(x), y_perm(y), ref_perm(ref), out_perm(ref), x_perm_row_major(x), y_perm_row_major(y), ref_perm_row_major(ref);
    for (int j = 0; j < mu; j++) {
        if (op == 'T' || op == 'C') {
            global_to_root_cluster(target_cluster_tree, x.data() + ni * j, x_perm.data() + ni * j);
            global_to_root_cluster(source_cluster_tree, y.data() + no * j, y_perm.data() + no * j);
            global_to_root_cluster(source_cluster_tree, ref.data() + no * j, ref_perm.data() + no * j);
        } else {
            global_to_root_cluster(source_cluster_tree, x.data() + ni * j, x_perm.data() + ni * j);
            global_to_root_cluster(target_cluster_tree, y.data() + no * j, y_perm.data() + no * j);
            global_to_root_cluster(target_cluster_tree, ref.data() + no * j, ref_perm.data() + no * j);
        }
    }

    for (int i = 0; i < ni; i++) {
        for (int j = 0; j < mu; j++) {
            x_perm_row_major[i * mu + j] = x_perm[i + j * ni];
        }
    }

    for (int i = 0; i < no; i++) {
        for (int j = 0; j < mu; j++) {
            y_perm_row_major[i * mu + j]   = y_perm[i + j * no];
            ref_perm_row_major[i * mu + j] = ref_perm[i + j * no];
        }
    }

    if (mu == 1) {
        out_perm = y_perm;
        A.add_vector_product_global_to_local(alpha, x_perm.data(), beta, out_perm.data());
        error    = norm2(ref_perm - out_perm) / norm2(ref_perm);
        is_error = is_error || !(error < Fixed_approximation.get_epsilon() * (1 + margin));
        cout << "> Errors on a matrix vector product with fixed approximation: " << error << endl;
    }

    // // Test
    // std::vector<double> in_permuted(size, 1);
    // std::vector<double> out(size, 1);
    // std::vector<double> out_permuted(size, 0);
    // std::vector<double> out_ref(size, 1);
    // generator.mvprod(in.data(), out_ref.data(), 1);

    // global_to_root_cluster(*cluster_tree, in.data(), in_permuted.data());
    // A.add_vector_product_global_to_local(1, in_permuted.size(), in_permuted.data(), 0, out_permuted.size(), out_permuted.data());
    // root_cluster_to_global(*cluster_tree, out_permuted.data(), out.data());

    // // Error
    // double error = norm2(out - out_ref) / norm2(out_ref);
    // cout << "error: " << error << endl;
    // test = test || !(error < 1e-14);

    // MPI_Finalize();
    return test;
}
