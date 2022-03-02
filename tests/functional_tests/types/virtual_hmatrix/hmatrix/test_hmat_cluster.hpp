#include <complex>
#include <iostream>
#include <random>
#include <vector>

#include <htool/blocks/virtual_block_data.hpp>
#include <htool/clustering/cluster.hpp>
#include <htool/clustering/pca.hpp>
#include <htool/lrmat/SVD.hpp>
#include <htool/lrmat/fullACA.hpp>
#include <htool/lrmat/partialACA.hpp>
#include <htool/lrmat/sympartialACA.hpp>
#include <htool/testing/generator_test.hpp>
#include <htool/testing/geometry.hpp>
#include <htool/types/hmatrix.hpp>

using namespace std;
using namespace htool;

template <typename ClusterImpl, template <typename> class CompressionImpl>
int test_hmat_cluster(int argc, char *argv[], double margin = 0) {

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
    distance[0] = 10;
    distance[1] = 20;
    distance[2] = 30;
    distance[3] = 40;

    double epsilon     = 1e-8;
    double eta         = 0.1;
    int minclustersize = 10;
    int maxblocksize   = 1000000;
    int minsourcedepth = 0;
    int mintargetdepth = 0;

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
        GeneratorTestDouble A(3, nr, nc, p1, p2);
        std::shared_ptr<VirtualCluster> t = make_shared<ClusterImpl>();
        std::shared_ptr<VirtualCluster> s = make_shared<ClusterImpl>();
        t->build(nr, p1.data(), 2);
        s->build(nc, p2.data(), 2);

        std::shared_ptr<VirtualLowRankGenerator<double>> compressor = std::make_shared<CompressionImpl<double>>();
        HMatrix<double> HA(t, s, epsilon, eta);
        HA.set_epsilon(epsilon);
        HA.set_eta(eta);
        HA.set_compression(compressor);
        HA.set_minsourcedepth(minsourcedepth);
        HA.set_mintargetdepth(mintargetdepth);
        HA.set_mintargetdepth(mintargetdepth);
        HA.set_maxblocksize(maxblocksize);

        // Getters
        test = test || !(abs(HA.get_epsilon() - epsilon) < 1e-10);
        test = test || !(abs(HA.get_eta() - eta) < 1e-10);
        test = test || !(HA.get_minsourcedepth() == minsourcedepth);
        test = test || !(HA.get_mintargetdepth() == mintargetdepth);
        test = test || !(HA.get_maxblocksize() == maxblocksize);
        test = test || !(HA.get_dimension() == 1);
        test = test || !(HA.get_MasterOffset_s().size() == size);
        test = test || !(HA.get_MasterOffset_t().size() == size);

        HA.build(A, p1.data(), p2.data());
        HA.print_infos();
        auto info = HA.get_infos();

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
        HA.mvprod_global_to_global(f.data(), result.data(), 1);
        double erreur2 = norm2(A * f - result) / norm2(A * f);
        // double erreurFrob = Frobenius_absolute_error(HA.get(), A) / A.normFrob();

        // test = test || !(erreurFrob < (1 + margin) * HA,get_epsilon());
        test = test || !(erreur2 < HA.get_epsilon());

        if (rank == 0) {
            // cout << "Errors with Frobenius norm: " << erreurFrob << endl;
            cout << "Errors on a mat vec prod : " << erreur2 << endl;
            cout << "test: " << test << endl;
        }
    }

    return test;
}
