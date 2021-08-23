#include <htool/clustering/pca.hpp>
#include <htool/htool.hpp>
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

        int nr = 2000;
        int nc = 1000;

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

        int size_numbering = nr / size;
        int count_size     = 0;
        std::vector<int> MasterOffset_target;
        for (int p = 0; p < size - 1; p++) {
            MasterOffset_target.push_back(count_size);
            MasterOffset_target.push_back(size_numbering);

            count_size += size_numbering;
        }
        MasterOffset_target.push_back(count_size);
        MasterOffset_target.push_back(nr - count_size);

        size_numbering = nc / size;
        count_size     = 0;

        std::vector<int> MasterOffset_source;
        for (int p = 0; p < size - 1; p++) {
            MasterOffset_source.push_back(count_size);
            MasterOffset_source.push_back(size_numbering);

            count_size += size_numbering;
        }
        MasterOffset_source.push_back(count_size);
        MasterOffset_source.push_back(nc - count_size);

        // local clustering
        std::shared_ptr<Cluster<PCAGeometricClustering>> t = make_shared<Cluster<PCAGeometricClustering>>();
        std::shared_ptr<Cluster<PCAGeometricClustering>> s = make_shared<Cluster<PCAGeometricClustering>>();
        t->build(nr, p1.data(), MasterOffset_target.data(), 2);
        s->build(nc, p2.data(), MasterOffset_source.data(), 2);

        std::shared_ptr<fullACA<double>> compressor = std::make_shared<fullACA<double>>();
        HMatrix<double> HA(t, s, epsilon, eta);
        HA.set_compression(compressor);
        HA.build(A, p1.data(), p2.data());
        HA.print_infos();

        // Global vectors
        std::vector<double> x_global(nc, 1), f_global(nr), f_global_test(nr);
        f_global = A * x_global;

        // Global product
        HA.mvprod_global_to_global(x_global.data(), f_global_test.data());

        // Errors
        double global_diff = norm2(f_global - f_global_test) / norm2(f_global);

        if (rank == 0) {
            cout << "difference on mat vec prod computed globally: " << global_diff << endl;
        }
        test = test || !(global_diff < HA.get_epsilon());

        // Local vectors
        std::vector<double> x_local(MasterOffset_source[2 * rank + 1], 1), f_local(MasterOffset_target[2 * rank + 1]), f_local_to_global(nr);

        // Local product
        HA.mvprod_local_to_local(x_local.data(), f_local.data());
        HA.local_to_global_target(f_local.data(), f_local_to_global.data(), 1);

        // Error
        double global_local_diff = norm2(f_global_test - f_local_to_global) / norm2(f_global_test);

        if (rank == 0) {
            cout << "difference on mat vec prod computed globally and locally: " << global_local_diff << endl;
        }
        test = test || !(global_local_diff < 1e-10);
    }
    if (rank == 0) {
        cout << "test: " << test << endl;
    }
    // Finalize the MPI environment.
    MPI_Finalize();

    return test;
}
