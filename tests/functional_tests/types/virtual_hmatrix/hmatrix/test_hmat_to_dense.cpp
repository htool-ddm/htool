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
    bool test = 0;

    double distance    = 1;
    double epsilon     = 1e-6;
    double eta         = 0.1;
    int minclustersize = 2;

    srand(1);
    // we set a constant seed for rand because we want always the same result if we run the check many times
    // (two different initializations with the same seed will generate the same succession of results in the subsequent calls to rand)

    int nr = 1000;
    int nc = 500;

    double z1 = 1;
    double z2 = 1 + distance;
    vector<double> p1(3 * nr);
    vector<double> p2(3 * nc);

    srand(1);
    // we set a constant seed for rand because we want always the same result if we run the check many times
    // (two different initializations with the same seed will generate the same succession of results in the subsequent calls to rand)
    create_disk(3, z1, nr, p1.data());
    create_disk(3, z2, nc, p2.data());

    GeneratorTestDouble A(3, nr, nc, p1, p2);

    // Hmatrix
    std::shared_ptr<Cluster<PCAGeometricClustering>> t = make_shared<Cluster<PCAGeometricClustering>>();
    std::shared_ptr<Cluster<PCAGeometricClustering>> s = make_shared<Cluster<PCAGeometricClustering>>();
    t->build(nr, p1.data());
    s->build(nc, p2.data());
    t->set_minclustersize(minclustersize);
    s->set_minclustersize(minclustersize);
    std::shared_ptr<fullACA<double>> compressor = std::make_shared<fullACA<double>>();
    HMatrix<double> HA(t, s, epsilon, eta);
    HA.set_compression(compressor);
    HA.build(A, p1.data(), p2.data());
    HA.print_infos();

    // Dense Matrix
    Matrix<double> DA_local = HA.get_local_dense();

    // Test dense matrices
    double error = 0;
    double norm  = 0;
    for (int i = 0; i < t->get_local_size(); i++) {
        for (int j = 0; j < nc; j++) {
            error += std::abs((A.get_coef(t->get_global_perm(i + t->get_local_offset()), s->get_global_perm(j)) - DA_local(i, j)) * (A.get_coef(t->get_global_perm(i + t->get_local_offset()), s->get_global_perm(j)) - DA_local(i, j)));
            norm += std::abs(A.get_coef(t->get_global_perm(i + t->get_local_offset()), s->get_global_perm(j)) * A.get_coef(t->get_global_perm(i + t->get_local_offset()), s->get_global_perm(j)));
        }
    }
    std::cout << error << std::endl;
    if (rank == 0) {
        cout << "Difference between dense matrix and local dense matrix: " << std::sqrt(error / norm) << endl;
    }
    test = test || !(std::sqrt(error / norm) < epsilon);

    // Finalize the MPI environment.
    MPI_Finalize();

    return test;
}
