#include <htool/clustering/pca.hpp>
#include <htool/htool.hpp>
#include <htool/lrmat/fullACA.hpp>
#include <htool/testing/generator_test.hpp>
#include <htool/testing/geometry.hpp>
#include <htool/types/hmatrix.hpp>

using namespace std;
using namespace htool;

int test_hmat_local_get(int argc, char *argv[], char symmetry, char UPLO) {

    // Get the number of processes
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Get the rank of the process
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    //
    bool test      = 0;
    double epsilon = 1e-6;
    double eta     = 0.1;

    srand(rank);
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
    std::vector<int> MasterOffset;
    for (int p = 0; p < size - 1; p++) {
        MasterOffset.push_back(count_size);
        MasterOffset.push_back(size_numbering);

        count_size += size_numbering;
    }
    MasterOffset.push_back(count_size);
    MasterOffset.push_back(nr - count_size);

    // local clustering
    std::shared_ptr<Cluster<PCARegularClustering>> t = make_shared<Cluster<PCARegularClustering>>();
    t->build(nr, p1.data(), MasterOffset.data(), 2);

    HMatrix<double> HA(t, t, epsilon, eta, symmetry, UPLO);
    HA.build(A, p1.data());
    HA.print_infos();

    // Local permutation
    std::vector<int> local_perm_source = HA.get_local_perm_source();
    std::vector<int> local_perm_target = HA.get_local_perm_target();
    std::vector<int> perm_source       = HA.get_perms();
    std::vector<int> perm_target       = HA.get_permt();

    for (int i = 0; i < MasterOffset[rank * 2 + 1]; i++) {
        test = test || !(local_perm_source[i] == perm_source[MasterOffset[rank * 2] + i]);
        test = test || !(local_perm_target[i] == perm_target[MasterOffset[rank * 2] + i]);
    }

    // Local diagonal
    std::vector<double> local_diagonal = HA.get_local_diagonal(true);

    double error = 0;
    double norm  = 0;
    for (int i = 0; i < MasterOffset[2 * rank + 1]; i++) {
        error += (A.get_coef(i + MasterOffset[2 * rank], i + MasterOffset[2 * rank]) - local_diagonal[i]) * (A.get_coef(i + MasterOffset[2 * rank], i + MasterOffset[2 * rank]) - local_diagonal[i]);
        ;
        norm += A.get_coef(i + MasterOffset[2 * rank], i + MasterOffset[2 * rank]) * A.get_coef(i + MasterOffset[2 * rank], i + MasterOffset[2 * rank]);
    }

    if (rank == 0) {
        cout << "Error on local diagonal: " << std::sqrt(error / norm) << endl;
    }
    test = test || !(std::sqrt(error / norm) < 1e-10);

    // Local block
    Matrix<double> local_diagonal_block = HA.get_local_diagonal_block(true);

    error = 0;
    norm  = 0;
    for (int j = 0; j < MasterOffset[2 * rank + 1]; j++) {
        for (int i = 0; i < MasterOffset[2 * rank + 1]; i++) {
            error += (A.get_coef(i + MasterOffset[2 * rank], j + MasterOffset[2 * rank]) - local_diagonal_block(i, j)) * (A.get_coef(i + MasterOffset[2 * rank], j + MasterOffset[2 * rank]) - local_diagonal_block(i, j));
            norm += A.get_coef(i + MasterOffset[2 * rank], j + MasterOffset[2 * rank]) * A.get_coef(i + MasterOffset[2 * rank], j + MasterOffset[2 * rank]);
        }
    }

    if (rank == 0) {
        cout << "Error on local diagonal block: " << std::sqrt(error / norm) << endl;
    }
    test = test || !(std::sqrt(error / norm) < HA.get_epsilon());

    if (rank == 0) {
        cout << "test: " << test << " for " << symmetry << " " << UPLO << endl;
    }

    return test;
}

int test_hmat_local_get_complex(int argc, char *argv[], char symmetry, char UPLO) {

    // Get the number of processes
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Get the rank of the process
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    //
    bool test      = 0;
    double epsilon = 1e-6;
    double eta     = 0.1;

    srand(rank);
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
    std::vector<int> MasterOffset;
    for (int p = 0; p < size - 1; p++) {
        MasterOffset.push_back(count_size);
        MasterOffset.push_back(size_numbering);

        count_size += size_numbering;
    }
    MasterOffset.push_back(count_size);
    MasterOffset.push_back(nr - count_size);

    // local clustering
    std::shared_ptr<Cluster<PCAGeometricClustering>> t = make_shared<Cluster<PCAGeometricClustering>>();
    t->build(nr, p1.data(), MasterOffset.data(), 2);

    HMatrix<std::complex<double>> HA(t, t, epsilon, eta, symmetry, UPLO);
    HA.build(A, p1.data());
    HA.print_infos();

    // Local diagonal
    std::vector<std::complex<double>> local_diagonal = HA.get_local_diagonal(true);

    double error = 0;
    double norm  = 0;
    for (int i = 0; i < MasterOffset[2 * rank + 1]; i++) {
        error += std::abs((A.get_coef(i + MasterOffset[2 * rank], i + MasterOffset[2 * rank]) - local_diagonal[i]) * (A.get_coef(i + MasterOffset[2 * rank], i + MasterOffset[2 * rank]) - local_diagonal[i]));
        ;
        norm += std::abs(A.get_coef(i + MasterOffset[2 * rank], i + MasterOffset[2 * rank]) * A.get_coef(i + MasterOffset[2 * rank], i + MasterOffset[2 * rank]));
    }

    if (rank == 0) {
        cout << "Error on local diagonal: " << std::sqrt(error / norm) << endl;
    }
    test = test || !(std::sqrt(error / norm) < 1e-10);

    // Local block
    Matrix<std::complex<double>> local_diagonal_block = HA.get_local_diagonal_block(true);

    error = 0;
    norm  = 0;
    for (int j = 0; j < MasterOffset[2 * rank + 1]; j++) {
        for (int i = 0; i < MasterOffset[2 * rank + 1]; i++) {
            error += std::abs((A.get_coef(i + MasterOffset[2 * rank], j + MasterOffset[2 * rank]) - local_diagonal_block(i, j)) * (A.get_coef(i + MasterOffset[2 * rank], j + MasterOffset[2 * rank]) - local_diagonal_block(i, j)));
            norm += std::abs(A.get_coef(i + MasterOffset[2 * rank], j + MasterOffset[2 * rank]) * A.get_coef(i + MasterOffset[2 * rank], j + MasterOffset[2 * rank]));
        }
    }

    if (rank == 0) {
        cout << "Error on local diagonal block: " << std::sqrt(error / norm) << endl;
    }
    test = test || !(std::sqrt(error / norm) < HA.get_epsilon());

    if (rank == 0) {
        cout << "test: " << test << " for " << symmetry << " " << UPLO << endl;
    }

    return test;
}

int test_hmat_local_get_complex_hermitian(int argc, char *argv[], char symmetry, char UPLO) {

    // Get the number of processes
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Get the rank of the process
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    //
    bool test      = 0;
    double epsilon = 1e-6;
    double eta     = 0.1;

    srand(rank);
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
    std::vector<int> MasterOffset;
    for (int p = 0; p < size - 1; p++) {
        MasterOffset.push_back(count_size);
        MasterOffset.push_back(size_numbering);

        count_size += size_numbering;
    }
    MasterOffset.push_back(count_size);
    MasterOffset.push_back(nr - count_size);

    // local clustering
    std::shared_ptr<Cluster<PCAGeometricClustering>> t = make_shared<Cluster<PCAGeometricClustering>>();
    t->build(nr, p1.data(), MasterOffset.data(), 2);

    HMatrix<std::complex<double>> HA(t, t, epsilon, eta, symmetry, UPLO);
    HA.build(A, p1.data());
    HA.print_infos();

    // Local diagonal
    std::vector<std::complex<double>> local_diagonal = HA.get_local_diagonal(true);

    double error = 0;
    double norm  = 0;
    for (int i = 0; i < MasterOffset[2 * rank + 1]; i++) {
        error += std::abs((A.get_coef(i + MasterOffset[2 * rank], i + MasterOffset[2 * rank]) - local_diagonal[i]) * (A.get_coef(i + MasterOffset[2 * rank], i + MasterOffset[2 * rank]) - local_diagonal[i]));
        ;
        norm += std::abs(A.get_coef(i + MasterOffset[2 * rank], i + MasterOffset[2 * rank]) * A.get_coef(i + MasterOffset[2 * rank], i + MasterOffset[2 * rank]));
    }

    if (rank == 0) {
        cout << "Error on local diagonal: " << std::sqrt(error / norm) << endl;
    }
    test = test || !(std::sqrt(error / norm) < 1e-10);

    // Local block
    Matrix<std::complex<double>> local_diagonal_block = HA.get_local_diagonal_block(true);

    error = 0;
    norm  = 0;
    for (int j = 0; j < MasterOffset[2 * rank + 1]; j++) {
        for (int i = 0; i < MasterOffset[2 * rank + 1]; i++) {
            error += std::abs((A.get_coef(i + MasterOffset[2 * rank], j + MasterOffset[2 * rank]) - local_diagonal_block(i, j)) * (A.get_coef(i + MasterOffset[2 * rank], j + MasterOffset[2 * rank]) - local_diagonal_block(i, j)));
            norm += std::abs(A.get_coef(i + MasterOffset[2 * rank], j + MasterOffset[2 * rank]) * A.get_coef(i + MasterOffset[2 * rank], j + MasterOffset[2 * rank]));
        }
    }

    if (rank == 0) {
        cout << "Error on local diagonal block: " << std::sqrt(error / norm) << endl;
    }
    test = test || !(std::sqrt(error / norm) < HA.get_epsilon());

    if (rank == 0) {
        cout << "test: " << test << " for " << symmetry << " " << UPLO << endl;
    }

    return test;
}
