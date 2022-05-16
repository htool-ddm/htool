#include <complex>
#include <iostream>
#include <vector>

#include <htool/clustering/pca.hpp>
#include <htool/lrmat/fullACA.hpp>
#include <htool/testing/generator_test.hpp>
#include <htool/testing/geometry.hpp>
#include <htool/types/hmatrix.hpp>
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
    bool test           = 0;
    const int ndistance = 4;
    double distance[ndistance];
    distance[0] = 1;
    distance[1] = 2;
    distance[2] = 3;
    distance[3] = 4;

    double epsilon = 1e-8;
    double eta     = 0.1;

    for (int idist = 0; idist < ndistance; idist++) {
        // cout << "Distance between the clusters: " << distance[idist] << endl;

        srand(1);
        // we set a constant seed for rand because we want always the same result if we run the check many times
        // (two different initializations with the same seed will generate the same succession of results in the subsequent calls to rand)

        int nr = 500;
        int nc = 400;

        double z1 = 1;
        double z2 = 1 + distance[idist];
        vector<double> p1(3 * nr);
        vector<double> p2(3 * nc);

        srand(1);
        // we set a constant seed for rand because we want always the same result if we run the check many times
        // (two different initializations with the same seed will generate the same succession of results in the subsequent calls to rand)
        create_disk(3, z1, nr, p1.data());
        create_disk(3, z2, nc, p2.data());

        GeneratorTestDouble A(3, nr, nc, p1, p2);
        std::shared_ptr<Cluster<PCAGeometricClustering>> t = make_shared<Cluster<PCAGeometricClustering>>();
        std::shared_ptr<Cluster<PCAGeometricClustering>> s = make_shared<Cluster<PCAGeometricClustering>>();
        t->build(nr, p1.data());
        s->build(nc, p2.data());
        std::shared_ptr<fullACA<double>> compressor = std::make_shared<fullACA<double>>();
        HMatrix<double> HA(t, s, epsilon, eta);
        HA.set_compression(compressor);
        HA.build(A, p1.data(), p2.data());
        HA.print_infos();
        auto test = HA.get_output();
        HA.save_plot("plot_" + NbrToStr(idist));
    }

    // Finalize the MPI environment.
    MPI_Finalize();
    return test;
}
