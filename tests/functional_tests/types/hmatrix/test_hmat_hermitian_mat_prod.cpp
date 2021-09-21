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
    int mu      = 5;

    double epsilon = 1e-3;
    double eta     = 0.1;

    for (int idist = 0; idist < ndistance; idist++) {

        srand(1);
        // we set a constant seed for rand because we want always the same result if we run the check many times
        // (two different initializations with the same seed will generate the same succession of results in the subsequent calls to rand)

        int nr    = 500;
        int nc    = 500;
        double z1 = 1;
        vector<double> p1(3 * nr);
        srand(1);
        // we set a constant seed for rand because we want always the same result if we run the check many times
        // (two different initializations with the same seed will generate the same succession of results in the subsequent calls to rand)
        create_disk(3, z1, nr, p1.data());
        GeneratorTestComplexHermitian A(3, nr, p1, 1);

        std::shared_ptr<Cluster<PCARegularClustering>> t = make_shared<Cluster<PCARegularClustering>>();
        t->build(nr, p1.data(), 2);

        std::shared_ptr<fullACA<std::complex<double>>> compressor = std::make_shared<fullACA<std::complex<double>>>();
        HMatrix<complex<double>> HA_L(t, t, epsilon, eta, 'H', 'L');
        HA_L.set_compression(compressor);
        HA_L.build(A, p1.data());
        HA_L.print_infos();

        HMatrix<std::complex<double>> HA_U(t, t, epsilon, eta, 'H', 'U');
        HA_U.set_compression(compressor);
        HA_U.build(A, p1.data());
        HA_U.print_infos();

        // Global vectors
        std::vector<complex<double>> x_global(nc * mu, std::complex<double>(2, 2)), f_global(nr * mu), f_global_L(nr * mu), f_global_U(nr * mu);
        A.mvprod(x_global.data(), f_global.data(), mu);

        // Global product
        HA_L.mvprod_global_to_global(x_global.data(), f_global_L.data(), mu);
        HA_U.mvprod_global_to_global(x_global.data(), f_global_U.data(), mu);

        // std::transform(f_global_U.begin(),f_global_U.end(),f_global_U.begin(),[](std::complex<double>&c){return std::conj(c);});
        // Errors
        double global_diff_L   = norm2(f_global - f_global_L) / norm2(f_global);
        double global_diff_U   = norm2(f_global - f_global_U) / norm2(f_global);
        double global_diff_L_U = norm2(f_global_L - f_global_U) / norm2(f_global);

        if (rank == 0) {
            cout << "difference on mat mat prod computed globally for lower hermitian matrix: " << global_diff_L << endl;
        }
        test = test || !(global_diff_L < HA_L.get_epsilon());

        if (rank == 0) {
            cout << "difference on mat mat prod computed globally for upper hermitian matrix: " << global_diff_U << endl;
        }
        test = test || !(global_diff_U < HA_U.get_epsilon());

        if (rank == 0) {
            cout << "difference on mat mat prod computed globally between upper hermitian matrix and lower hermitian matrix: " << global_diff_L_U << endl;
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
