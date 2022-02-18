#include <complex>
#include <iostream>
#include <random>
#include <vector>

#include <htool/clustering/cluster.hpp>
#include <htool/clustering/pca.hpp>
#include <htool/lrmat/fullACA.hpp>
#include <htool/testing/generator_test.hpp>
#include <htool/testing/geometry.hpp>
#include <htool/types/hmatrix.hpp>
#include <htool/types/virtual_dense_blocks_generator.hpp>

using namespace std;
using namespace htool;

class BlockGenerator : public VirtualDenseBlocksGenerator<double> {
  private:
    const VirtualGenerator<double> &mat;

  public:
    BlockGenerator(const VirtualGenerator<double> &mat0) : mat(mat0){};

    void copy_dense_blocks(const std::vector<int> &M, const std::vector<int> &N, const std::vector<const int *> &rows, const std::vector<const int *> &cols, std::vector<double *> &ptr) const override {
        for (int i = 0; i < M.size(); i++) {
            mat.copy_submatrix(M[i], N[i], rows[i], cols[i], ptr[i]);
        }
    };
};

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
    distance[0] = 10;
    distance[1] = 20;
    distance[2] = 30;
    distance[3] = 40;

    double epsilon     = 1e-8;
    double eta         = 0.1;
    int minclustersize = 10;
    int maxblocksize   = 1000;
    int minsourcedepth = 0;
    int mintargetdepth = 0;

    for (int idist = 0; idist < ndistance; idist++) {
        // cout << "Distance between the clusters: " << distance[idist] << endl;

        srand(1);
        // we set a constant seed for rand because we want always the same result if we run the check many times
        // (two different initializations with the same seed will generate the same succession of results in the subsequent calls to rand)

        int nr = 500;

        double z1 = 1;
        vector<double> p1(3 * nr);

        srand(1);
        // we set a constant seed for rand because we want always the same result if we run the check many times
        // (two different initializations with the same seed will generate the same succession of results in the subsequent calls to rand)
        create_disk(3, z1, nr, p1.data());

        vector<double> rhs(p1.size(), 1);
        GeneratorTestDoubleSymmetric A(3, nr, nr, p1, p1);
        std::shared_ptr<Cluster<PCARegularClustering>> t = make_shared<Cluster<PCARegularClustering>>();
        t->build(nr, p1.data(), 2);

        std::shared_ptr<fullACA<double>> compressor = std::make_shared<fullACA<double>>();
        HMatrix<double> HA(t, t, epsilon, eta);
        HA.set_epsilon(epsilon);
        HA.set_eta(eta);
        HA.set_compression(compressor);
        HA.set_minsourcedepth(minsourcedepth);
        HA.set_mintargetdepth(mintargetdepth);
        HA.set_mintargetdepth(mintargetdepth);
        HA.set_maxblocksize(maxblocksize);
        HA.set_delay_dense_computation(true);

        // Getters
        test = test || !(abs(HA.get_epsilon() - epsilon) < 1e-10);
        test = test || !(abs(HA.get_eta() - eta) < 1e-10);
        test = test || !(HA.get_minsourcedepth() == minsourcedepth);
        test = test || !(HA.get_mintargetdepth() == mintargetdepth);
        test = test || !(HA.get_maxblocksize() == maxblocksize);
        test = test || !(HA.get_dimension() == 1);
        test = test || !(HA.get_MasterOffset_s().size() == size);
        test = test || !(HA.get_MasterOffset_t().size() == size);

        HA.build(A, p1.data());
        HA.print_infos();
        auto info = HA.get_infos();

        // Delayed dense computations
        BlockGenerator blockgenerator(A);
        HA.build_dense_blocks(blockgenerator);

        // Random vector
        vector<double> f(nr, 1);
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
        MPI_Bcast(f.data(), nr, MPI_DOUBLE, 0, MPI_COMM_WORLD);

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

    // Finalize the MPI environment.
    MPI_Finalize();
    return test;
}
