#include "test_multi_hmat.hpp"
#include <htool/testing/generator_test.hpp>
#include <htool/testing/geometry.hpp>

using namespace std;
using namespace htool;

int main(int argc, char *argv[]) {

    // Initialize the MPI environment
    MPI_Init(&argc, &argv);

    // Get the number of processes
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Get the rank of the process
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    //
    const int ndistance = 4;
    double distance[ndistance];
    distance[0] = 15;
    distance[1] = 20;
    distance[2] = 30;
    distance[3] = 40;

    double epsilon = 1e-8;
    double eta     = 0.1;

    int nr = 500;
    int nc = 400;
    std::vector<double> xt(3 * nr);
    std::vector<double> xs(3 * nc);

    bool test = 0;

    //
    for (int idist = 0; idist < ndistance; idist++) {

        srand(1);
        create_disk(3, 0, nr, xt.data());
        create_disk(3, distance[idist], nc, xs.data());

        vector<double> rhs(xs.size(), 1);
        MyMultiMatrix MultiA(3, nr, nc, xt, xs);
        int nm = MultiA.nb_matrix();
        GeneratorTestDouble A(3, nr, nc, xt, xs);

        std::shared_ptr<Cluster<PCAGeometricClustering>> t = make_shared<Cluster<PCAGeometricClustering>>();
        std::shared_ptr<Cluster<PCAGeometricClustering>> s = make_shared<Cluster<PCAGeometricClustering>>();
        t->build(nr, xt.data(), 2);
        s->build(nc, xs.data(), 2);
        MultiHMatrix<double, MultipartialACA, RjasanowSteinbach> MultiHA(t, s, epsilon, eta);
        MultiHA.build_auto(MultiA, xt.data(), xs.data());
        HMatrix<double, partialACA, RjasanowSteinbach> HA(t, s, epsilon, eta);
        HA.build_auto(A, xt.data(), xs.data());

        // Comparison with HMatrix
        std::vector<double> one(nc, 1);
        double error = norm2(MultiHA[0] * one);
        // test         = test || !(error < 1e-10);
        // cout << "> Errors compared to HMatrix: " << error << endl;
        // for (int l = 0; l < nm; l++) {
        //     test = test || (test_multi_hmat_cluster(MultiA, MultiHA, l));
        // }
    }

    // Finalize the MPI environment.
    MPI_Finalize();

    return test;
}
