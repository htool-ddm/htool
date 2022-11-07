#include <htool/clustering/bounding_box_1.hpp>
#include <htool/clustering/cluster.hpp>
#include <htool/clustering/splitting.hpp>
#include <htool/testing/generator_test.hpp>
#include <htool/testing/geometry.hpp>
#include <htool/testing/local_dense_matrix.hpp>
#include <random>

using namespace std;
using namespace htool;

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    srand(1);
    bool test = 0;

    // Geometry
    int size = 200;
    int dim  = 3;
    vector<double> pt(size * dim);
    vector<double> r(size, 0);
    vector<double> g(size, 1);
    vector<int> tab(size);

    srand(1);
    // we set a constant seed for rand because we want always the same result if we run the check many times
    // (two different initializations with the same seed will generate the same succession of results in the subsequent calls to rand)
    create_disk(dim, 0, size, pt.data());

    // Cluster
    std::shared_ptr<VirtualCluster> cluster = make_shared<Cluster<BoundingBox1<SplittingTypes::GeometricSplitting>>>(dim);
    cluster->build(size, pt.data());

    // Generator
    GeneratorTestDoubleSymmetric generator(3, size, size, pt, pt);

    // LocalDenseMatrix
    LocalDenseMatrix<double> A(cluster, cluster);
    A.build(generator);

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
    std::vector<double> out_permuted(size, 1);
    std::vector<double> out_ref(size, 1);
    generator.mvprod(in.data(), out_ref.data(), 1);

    global_to_cluster(cluster.get(), in.data(), in_permuted.data());
    A.mvprod(in_permuted.data(), out_permuted.data());
    cluster_to_global(cluster.get(), out_permuted.data(), out.data());

    // Error
    double error = norm2(out - out_ref) / norm2(out_ref);
    cout << "error: " << error << endl;
    test = test || !(error < 1e-14);

    MPI_Finalize();
    return test;
}
