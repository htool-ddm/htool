#include <htool/clustering/clustering.hpp>
#include <htool/local_operators/local_dense_matrix.hpp>
#include <htool/testing/generator_test.hpp>
#include <htool/testing/geometry.hpp>
#include <mpi.h>
#include <random>

using namespace std;
using namespace htool;

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int sizeWorld;
    MPI_Comm_size(MPI_COMM_WORLD, &sizeWorld);
    srand(1);
    bool test = 0;

    // Geometry
    int size = 200;
    int dim  = 3;
    vector<double> pt(size * dim);
    vector<double> r(size, 0);
    vector<double> g(size, 1);
    vector<int> tab(size);

    double z1 = 1;
    create_disk(dim, z1, size, pt.data());

    // Cluster
    ClusterTreeBuilder<double, ComputeLargestExtent<double>, RegularSplitting<double>> recursive_build_strategy(size, dim, pt.data(), r.data(), g.data(), 2, sizeWorld);

    std::shared_ptr<Cluster<double>> cluster = make_shared<Cluster<double>>(recursive_build_strategy.create_cluster_tree());

    // Generator
    GeneratorTestDoubleSymmetric generator(3, size, size, pt, pt, cluster, cluster);

    // LocalDenseMatrix
    LocalDenseMatrix<double> A(generator, cluster, cluster);

    // Random input vector
    std::vector<double> in(size, 1);

    double lower_bound = 0;
    double upper_bound = 10000;
    std::random_device rd;
    std::mt19937 mersenne_engine(rd());
    std::uniform_real_distribution<double> dist(lower_bound, upper_bound);
    auto gen = [&dist, &mersenne_engine]() {
        return dist(mersenne_engine);
    };
    generate(begin(in), end(in), gen);

    // Test
    std::vector<double> in_permuted(size, 1);
    std::vector<double> out(size, 1);
    std::vector<double> out_permuted(size, 0);
    std::vector<double> out_ref(size, 1);
    generator.mvprod(in.data(), out_ref.data(), 1);

    global_to_root_cluster(*cluster, in.data(), in_permuted.data());
    A.add_vector_product_global_to_local(1, in_permuted.data(), 0, out_permuted.data());
    root_cluster_to_global(*cluster, out_permuted.data(), out.data());

    // Error
    double error = norm2(out - out_ref) / norm2(out_ref);
    cout << "error: " << error << endl;
    test = test || !(error < 1e-14);

    MPI_Finalize();
    return test;
}
