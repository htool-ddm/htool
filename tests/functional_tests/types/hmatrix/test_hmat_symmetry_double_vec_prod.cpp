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
    distance[0] = 3;
    distance[1] = 5;
    distance[2] = 7;
    distance[3] = 10;

    double epsilon = 1e-3;
    double eta     = 0.1;

    for (int idist = 0; idist < ndistance; idist++) {

        srand(1);
        // we set a constant seed for rand because we want always the same result if we run the check many times
        // (two different initializations with the same seed will generate the same succession of results in the subsequent calls to rand)

        int nr = 500;
        int nc = 500;

        double z1 = 1;
        vector<double> p1(3 * nr);

        srand(1);
        // we set a constant seed for rand because we want always the same result if we run the check many times
        // (two different initializations with the same seed will generate the same succession of results in the subsequent calls to rand)
        create_disk(3, z1, nr, p1.data());

        GeneratorTestDouble A(3, nr, p1, 1);

        std::shared_ptr<Cluster<PCARegularClustering>> t = make_shared<Cluster<PCARegularClustering>>();
        t->build(nr, p1.data(), 2);

        std::shared_ptr<fullACA<double>> compressor = std::make_shared<fullACA<double>>();

        HMatrix<double> HA_L(t, t, epsilon, eta, 'S', 'L');
        HA_L.set_compression(compressor);
        HA_L.build(A, p1.data());
        HA_L.print_infos();

        HMatrix<double> HA_U(t, t, epsilon, eta, 'S', 'U');
        HA_U.set_compression(compressor);
        HA_U.build(A, p1.data());
        HA_U.print_infos();

        // Global vectors
        std::vector<double> x_global(nc, 1), f_global(nr), f_global_L(nr), f_global_U(nr);
        f_global = A * x_global;

        // Global product
        HA_L.mvprod_global_to_global(x_global.data(), f_global_L.data());
        HA_U.mvprod_global_to_global(x_global.data(), f_global_U.data());

        // Errors
        double global_diff_L   = norm2(f_global - f_global_L) / norm2(f_global);
        double global_diff_U   = norm2(f_global - f_global_U) / norm2(f_global);
        double global_diff_L_U = norm2(f_global_U - f_global_U) / norm2(f_global);

        if (rank == 0) {
            cout << "difference on mat vec prod computed globally for lower symmetric matrix: " << global_diff_L << endl;
        }
        test = test || !(global_diff_L < HA_L.get_epsilon());

        if (rank == 0) {
            cout << "difference on mat vec prod computed globally for upper symmetric matrix: " << global_diff_U << endl;
        }
        test = test || !(global_diff_U < HA_U.get_epsilon());

        if (rank == 0) {
            cout << "difference on mat vec prod computed globally between upper symmetric matrix and lower symmetric matrix: " << global_diff_L_U << endl;
        }
        test = test || !(global_diff_L_U < 1e-10);
    }
    if (rank == 0) {
        cout << "test: " << test << endl;
    }
    // Finalize the MPI environment.
    MPI_Finalize();

    return test;
}
