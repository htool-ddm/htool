#include <complex>
#include <iostream>
#include <random>
#include <vector>

#include <htool/blocks/admissibility_conditions.hpp>
#include <htool/clustering/cluster.hpp>
#include <htool/clustering/ncluster.hpp>
#include <htool/lrmat/SVD.hpp>
#include <htool/lrmat/fullACA.hpp>
#include <htool/lrmat/partialACA.hpp>
#include <htool/lrmat/sympartialACA.hpp>
#include <htool/testing/geometry.hpp>
#include <htool/testing/imatrix_test.hpp>
#include <htool/types/hmatrix.hpp>

using namespace std;
using namespace htool;

template <typename ClusterImpl, template <typename, typename> class LowRankMatrix>
int test_hmat_auto(int argc, char *argv[], double margin = 0) {

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
    distance[0]    = 10;
    distance[1]    = 20;
    distance[2]    = 30;
    distance[3]    = 40;
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

        vector<double> rhs(p2.size(), 1);
        IMatrixTestDouble A(3, nr, nc, p1, p2);
        HMatrix<double, LowRankMatrix, ClusterImpl, RjasanowSteinbach> HA(3, epsilon, eta);
        HA.build_auto(A, p1.data(), p2.data());
        HA.print_infos();

        // Random vector
        vector<double> f(nc, 1);
        if (rank == 0) {
            double lower_bound = 0;
            double upper_bound = 10000;
            std::random_device rd;
            std::mt19937 mersenne_engine(rd());
            std::uniform_real_distribution<double> dist(lower_bound, upper_bound);
            auto gen = [&dist, &mersenne_engine]() {
                return dist(mersenne_engine);
            };

            generate(begin(f), end(f), gen);
        }
        MPI_Bcast(f.data(), nc, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        std::vector<double> result(nr, 0);
        result            = HA * f;
        double erreur2    = norm2(A * f - result) / norm2(A * f);
        double erreurFrob = Frobenius_absolute_error(HA, A) / A.normFrob();

        test = test || !(erreurFrob < (margin + 1) * HA.get_epsilon());
        test = test || !(erreur2 < HA.get_epsilon());

        if (rank == 0) {
            cout << "Errors with Frobenius norm: " << erreurFrob << endl;
            cout << "Errors on a mat vec prod : " << erreur2 << endl;
            cout << "test: " << test << endl
                 << endl;
        }
    }

    // Finalize the MPI environment.
    MPI_Finalize();
    return test;
}
