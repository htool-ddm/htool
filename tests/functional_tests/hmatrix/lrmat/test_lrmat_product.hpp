
#include <htool/clustering/clustering.hpp>
#include <htool/hmatrix/lrmat/lrmat.hpp>
#include <htool/testing/generator_input.hpp>
#include <htool/testing/generator_test.hpp>
#include <htool/testing/geometry.hpp>

using namespace std;
using namespace htool;

template <typename T, typename GeneratorTestType, class Compressor>
bool test_lrmat_product(int nr, int nc, int mu, char op, Compressor &compressor, htool::underlying_type<T> margin) {
    const int ndistance = 4;
    htool::underlying_type<T> distance[ndistance];
    distance[0] = 15;
    distance[1] = 20;
    distance[2] = 30;
    distance[3] = 40;

    htool::underlying_type<T> epsilon = 0.0001;
    std::vector<htool::underlying_type<T>> xt(3 * nr);
    std::vector<htool::underlying_type<T>> xs(3 * nc);
    std::vector<int> tabt(500);
    std::vector<int> tabs(100);
    bool is_error = 0;
    for (int idist = 0; idist < ndistance; idist++) {

        create_disk(3, 0., nr, xt.data());
        create_disk(3, distance[idist], nc, xs.data());

        ClusterTreeBuilder<htool::underlying_type<T>> recursive_build_strategy;
        Cluster<htool::underlying_type<T>> target_root_cluster = recursive_build_strategy.create_cluster_tree(nr, 3, xt.data(), 2, 2);
        Cluster<htool::underlying_type<T>> source_root_cluster = recursive_build_strategy.create_cluster_tree(nc, 3, xs.data(), 2, 2);

        GeneratorTestType A(3, nr, nc, xt, xs, target_root_cluster, source_root_cluster, true, true);

        // partialACA fixed rank
        int reqrank_max = 10;
        LowRankMatrix<T> Fixed_approximation(A, compressor, target_root_cluster, source_root_cluster, reqrank_max, epsilon);

        // ACA automatic building
        LowRankMatrix<T> Auto_approximation(A, compressor, target_root_cluster, source_root_cluster, -1, epsilon);

        // Input sizes
        int ni = (op == 'T' || op == 'C') ? nr : nc;
        int no = (op == 'T' || op == 'C') ? nc : nr;

        // Random vector
        vector<T> x(ni * mu, 1), y(no * mu, 1), ref(no * mu, 1);
        T alpha, beta;
        htool::underlying_type<T> error;
        generate_random_vector(x);
        generate_random_vector(y);
        generate_random_scalar(alpha);
        generate_random_scalar(beta);
        if (op == 'N') {
            std::vector<T> temp(no * mu, 0);
            A.mvprod(x.data(), temp.data(), mu);
            ref = mult(alpha, temp) + mult(beta, y);
        } else {
            std::vector<T> temp(no * mu, 0);
            A.mvprod_transp(x.data(), temp.data(), mu);
            ref = mult(alpha, temp) + mult(beta, y);
        }

        // Permutation
        vector<T> x_perm(x), y_perm(y), ref_perm(ref), out_perm(ref), x_perm_row_major(x), y_perm_row_major(y), ref_perm_row_major(ref);
        for (int j = 0; j < mu; j++) {
            if (op == 'T' || op == 'C') {
                global_to_root_cluster(target_root_cluster, x.data() + ni * j, x_perm.data() + ni * j);
                global_to_root_cluster(source_root_cluster, y.data() + no * j, y_perm.data() + no * j);
                global_to_root_cluster(source_root_cluster, ref.data() + no * j, ref_perm.data() + no * j);
            } else {
                global_to_root_cluster(source_root_cluster, x.data() + ni * j, x_perm.data() + ni * j);
                global_to_root_cluster(target_root_cluster, y.data() + no * j, y_perm.data() + no * j);
                global_to_root_cluster(target_root_cluster, ref.data() + no * j, ref_perm.data() + no * j);
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

        // Tests for fixed rank
        if (mu == 1) {
            out_perm = y_perm;
            Fixed_approximation.add_vector_product(op, alpha, x_perm.data(), beta, out_perm.data());
            error    = norm2(ref_perm - out_perm) / norm2(ref_perm);
            is_error = is_error || !(error < Fixed_approximation.get_epsilon() * (1 + margin));
            cout << "> Errors on a matrix vector product with fixed approximation: " << error << endl;
        }

        out_perm = y_perm;
        Fixed_approximation.add_matrix_product(op, alpha, x_perm.data(), beta, out_perm.data(), mu);
        error    = norm2(ref_perm - out_perm) / norm2(ref_perm);
        is_error = is_error || !(error < Fixed_approximation.get_epsilon() * (1 + margin));
        cout << "> Errors on a matrix matrix product with fixed approximation: " << error << endl;

        out_perm = y_perm_row_major;
        Fixed_approximation.add_matrix_product_row_major(op, alpha, x_perm_row_major.data(), beta, out_perm.data(), mu);
        error    = norm2(ref_perm_row_major - out_perm) / norm2(ref_perm_row_major);
        is_error = is_error || !(error < Fixed_approximation.get_epsilon() * (1 + margin));
        cout << "> Errors on a matrix matrix product with fixed approximation and row major input: " << error << endl;

        // Tests for automatic rank
        if (mu == 1) {
            out_perm = y_perm;
            Auto_approximation.add_vector_product(op, alpha, x_perm.data(), beta, out_perm.data());
            error    = norm2(ref_perm - out_perm) / norm2(ref_perm);
            is_error = is_error || !(error < Auto_approximation.get_epsilon() * (1 + margin));
            cout << "> Errors on a matrix vector product with automatic approximation: " << error << endl;
        }

        out_perm = y_perm;
        Auto_approximation.add_matrix_product(op, alpha, x_perm.data(), beta, out_perm.data(), mu);
        error    = norm2(ref_perm - out_perm) / norm2(ref_perm);
        is_error = is_error || !(error < Auto_approximation.get_epsilon() * (1 + margin));
        cout << "> Errors on a matrix matrix product with automatic approximation: " << error << endl;

        out_perm = y_perm_row_major;
        Auto_approximation.add_matrix_product_row_major(op, alpha, x_perm_row_major.data(), beta, out_perm.data(), mu);
        error    = norm2(ref_perm_row_major - out_perm) / norm2(ref_perm_row_major);
        is_error = is_error || !(error < Fixed_approximation.get_epsilon() * (1 + margin));
        cout << "> Errors on a matrix matrix product with fixed approximation and row major input: " << error << endl;

        cout << "test : " << is_error << endl
             << endl;
    }
    return is_error;
}
