#include <complex>
#include <iostream>
#include <vector>

#include "test_lrmat.hpp"
#include <htool/clustering/pca.hpp>
#include <htool/lrmat/fullACA.hpp>

using namespace std;
using namespace htool;

int main(int argc, char *argv[]) {
    // Initialize the MPI environment
    MPI_Init(&argc, &argv);

    const int ndistance = 4;
    double distance[ndistance];
    distance[0] = 15;
    distance[1] = 20;
    distance[2] = 30;
    distance[3] = 40;

    double epsilon = 0.0001;
    int nr         = 500;
    int nc         = 100;
    std::vector<double> xt(3 * nr);
    std::vector<double> xs(3 * nc);
    std::vector<int> tabt(500);
    std::vector<int> tabs(100);
    bool test = 0;
    for (int idist = 0; idist < ndistance; idist++) {

        srand(1);
        // we set a constant seed for rand because we want always the same result if we run the check many times
        // (two different initializations with the same seed will generate the same succession of results in the subsequent calls to rand)
        create_disk(3, 0, nr, xt.data(), tabt.data());
        create_disk(3, distance[idist], nc, xs.data(), tabs.data());

        Cluster<PCAGeometricClustering> t, s;

        t.build(nr, xt.data());
        s.build(nc, xs.data());

        IMatrixTestDouble A(3, nr, nc, xt, xs);

        // fullACA fixed rank
        int reqrank_max = 10;
        fullACA<double> A_fullACA_fixed(t.get_perm(), s.get_perm(), reqrank_max, epsilon);
        A_fullACA_fixed.build(A);

        // ACA automatic building
        fullACA<double> A_fullACA(t.get_perm(), s.get_perm());
        A_fullACA.set_epsilon(epsilon);
        A_fullACA.build(A);

        std::pair<double, double> fixed_compression_interval(0.87, 0.89);
        std::pair<double, double> auto_compression_interval(0.95, 0.97);
        test = test_lrmat(A, A_fullACA_fixed, A_fullACA, t.get_perm(), s.get_perm(), fixed_compression_interval, auto_compression_interval);
    }
    cout << "test : " << test << endl;

    // Finalize the MPI environment.
    MPI_Finalize();
    return test;
}
