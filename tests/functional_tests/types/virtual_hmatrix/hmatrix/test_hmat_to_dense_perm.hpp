#include <htool/clustering/pca.hpp>
#include <htool/lrmat/fullACA.hpp>
#include <htool/testing/generator_test.hpp>
#include <htool/testing/geometry.hpp>
#include <htool/types/hmatrix.hpp>

using namespace std;
using namespace htool;

int test_hmat_to_dense_perm(int argc, char *argv[]) {

    // Get the number of processes
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Get the rank of the process
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    //
    bool test = 0;

    double epsilon     = 1e-6;
    double eta         = 0.1;
    int minclustersize = 2;

    srand(1);
    // we set a constant seed for rand because we want always the same result if we run the check many times
    // (two different initializations with the same seed will generate the same succession of results in the subsequent calls to rand)

    int nr = 500;
    int nc = 500;

    double z1 = 1;
    vector<double> p2(3 * nc);
    vector<double> p1(3 * nr);
    srand(1);
    // we set a constant seed for rand because we want always the same result if we run the check many times
    // (two different initializations with the same seed will generate the same succession of results in the subsequent calls to rand)
    create_disk(3, z1, nr, p1.data());
    create_disk(3, z1 + 1, nc, p2.data());

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

    // Hmatrix
    std::shared_ptr<Cluster<PCAGeometricClustering>> t = make_shared<Cluster<PCAGeometricClustering>>();
    std::shared_ptr<Cluster<PCAGeometricClustering>> s = make_shared<Cluster<PCAGeometricClustering>>();
    t->set_minclustersize(minclustersize);
    s->set_minclustersize(minclustersize);
    t->build(nr, p1.data(), MasterOffset_target.data());
    s->build(nc, p2.data(), MasterOffset_source.data());
    std::shared_ptr<fullACA<double>> compressor = std::make_shared<fullACA<double>>();
    HMatrix<double> HA(t, s, epsilon, eta);
    HA.set_compression(compressor);
    HA.build(A, p1.data(), p2.data());
    HA.print_infos();

    // Dense Matrix
    Matrix<double> DA_local = HA.get_local_dense_perm();

    // Test dense matrices
    double error = 0;
    double norm  = 0;
    for (int i = 0; i < t->get_local_size(); i++) {
        for (int j = 0; j < nc; j++) {
            error += std::abs((DA_local(i, j) - A.get_coef(i + t->get_local_offset(), j)) * (DA_local(i, j) - A.get_coef(i + t->get_local_offset(), j)));
            norm += std::abs(A.get_coef(i + t->get_local_offset(), j) * A.get_coef(i + t->get_local_offset(), j));
        }
    }

    if (rank == 0) {
        cout << "Difference between dense matrix and local dense matrix: " << std::sqrt(error / norm) << endl;
    }
    test = test || !(std::sqrt(error / norm) < epsilon);

    if (rank == 0) {
        cout << "test: " << test << " for " << 'N' << " " << 'N' << endl;
    }
    return test;
}

int test_hmat_to_dense_perm_sym(int argc, char *argv[], char UPLO) {

    // Get the number of processes
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Get the rank of the process
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    //
    bool test = 0;

    double epsilon     = 1e-6;
    double eta         = 0.1;
    int minclustersize = 2;

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

    GeneratorTestDoubleSymmetric A(3, nr, nr, p1, p1);

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

    // Hmatrix
    std::shared_ptr<Cluster<PCAGeometricClustering>> t = make_shared<Cluster<PCAGeometricClustering>>();
    t->set_minclustersize(minclustersize);
    t->build(nr, p1.data(), MasterOffset_target.data());
    std::shared_ptr<fullACA<double>> compressor = std::make_shared<fullACA<double>>();
    HMatrix<double> HA(t, t, epsilon, eta, 'S', UPLO);
    HA.set_compression(compressor);
    HA.build(A, p1.data());
    HA.print_infos();

    // Dense Matrix
    Matrix<double> DA_local = HA.get_local_dense_perm();

    // Test dense matrices
    double error = 0;
    double norm  = 0;
    for (int i = 0; i < t->get_local_size(); i++) {
        for (int j = 0; j < nr; j++) {
            error += std::abs((DA_local(i, j) - A.get_coef(i + t->get_local_offset(), j)) * (DA_local(i, j) - A.get_coef(i + t->get_local_offset(), j)));
            norm += std::abs(A.get_coef(i + t->get_local_offset(), j) * A.get_coef(i + t->get_local_offset(), j));
        }
    }

    if (rank == 0) {
        cout << "Difference between dense matrix and local dense matrix: " << std::sqrt(error / norm) << endl;
    }
    test = test || !(std::sqrt(error / norm) < epsilon);

    if (rank == 0) {
        cout << "test: " << test << " for " << 'S' << " " << UPLO << endl;
    }

    return test;
}

int test_hmat_to_dense_perm_sym_complex(int argc, char *argv[], char UPLO) {

    // Get the number of processes
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Get the rank of the process
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    //
    bool test = 0;

    double epsilon     = 1e-6;
    double eta         = 0.1;
    int minclustersize = 2;

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

    GeneratorTestComplexSymmetric A(3, nr, nr, p1, p1);

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

    // Hmatrix
    std::shared_ptr<Cluster<PCAGeometricClustering>> t = make_shared<Cluster<PCAGeometricClustering>>();
    t->set_minclustersize(minclustersize);
    t->build(nr, p1.data(), MasterOffset_target.data());
    std::shared_ptr<fullACA<std::complex<double>>> compressor = std::make_shared<fullACA<std::complex<double>>>();
    HMatrix<std::complex<double>> HA(t, t, epsilon, eta, 'S', UPLO);
    HA.set_compression(compressor);
    HA.build(A, p1.data());
    HA.print_infos();

    // Dense Matrix
    Matrix<std::complex<double>> DA_local = HA.get_local_dense_perm();

    // Test dense matrices
    double error = 0;
    double norm  = 0;
    for (int i = 0; i < t->get_local_size(); i++) {
        for (int j = 0; j < nr; j++) {
            error += std::abs((DA_local(i, j) - A.get_coef(i + t->get_local_offset(), j)) * (DA_local(i, j) - A.get_coef(i + t->get_local_offset(), j)));
            norm += std::abs(A.get_coef(i + t->get_local_offset(), j) * A.get_coef(i + t->get_local_offset(), j));
        }
    }

    if (rank == 0) {
        cout << "Difference between dense matrix and local dense matrix: " << std::sqrt(error / norm) << endl;
    }
    test = test || !(std::sqrt(error / norm) < epsilon);

    if (rank == 0) {
        cout << "test: " << test << " for " << 'S' << " " << UPLO << endl;
    }

    return test;
}

int test_hmat_to_dense_perm_hermitian_complex(int argc, char *argv[], char UPLO) {

    // Get the number of processes
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Get the rank of the process
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    //
    bool test = 0;

    double epsilon     = 1e-6;
    double eta         = 0.1;
    int minclustersize = 2;

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

    GeneratorTestComplexHermitian A(3, nr, nr, p1, p1);

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

    // Hmatrix
    std::shared_ptr<Cluster<PCAGeometricClustering>> t = make_shared<Cluster<PCAGeometricClustering>>();
    t->set_minclustersize(minclustersize);
    t->build(nr, p1.data(), MasterOffset_target.data());
    std::shared_ptr<fullACA<std::complex<double>>> compressor = std::make_shared<fullACA<std::complex<double>>>();
    HMatrix<std::complex<double>> HA(t, t, epsilon, eta, 'H', UPLO);
    HA.set_compression(compressor);
    HA.build(A, p1.data());
    HA.print_infos();

    // Dense Matrix
    Matrix<std::complex<double>> DA_local = HA.get_local_dense_perm();

    // Test dense matrices
    double error = 0;
    double norm  = 0;
    for (int i = 0; i < t->get_local_size(); i++) {
        for (int j = 0; j < nr; j++) {
            error += std::abs((DA_local(i, j) - A.get_coef(i + t->get_local_offset(), j)) * (DA_local(i, j) - A.get_coef(i + t->get_local_offset(), j)));
            norm += std::abs(A.get_coef(i + t->get_local_offset(), j) * A.get_coef(i + t->get_local_offset(), j));
        }
    }

    if (rank == 0) {
        cout << "Difference between dense matrix and local dense matrix: " << std::sqrt(error / norm) << endl;
    }
    test = test || !(std::sqrt(error / norm) < epsilon);

    if (rank == 0) {
        cout << "test: " << test << " for " << 'H' << " " << UPLO << endl;
    }

    return test;
}
