#include "test_hmat_zero.hpp"

int main(int argc, char *argv[]) {
    // Initialize the MPI environment
    MPI_Init(&argc, &argv);

    bool test = test_hmat_zero<Cluster<PCA<SplittingTypes::GeometricSplitting>>, SVD>(argc, argv);

    test = test || test_hmat_zero<Cluster<PCA<SplittingTypes::GeometricSplitting>>, fullACA>(argc, argv);
    test = test || test_hmat_zero<Cluster<PCA<SplittingTypes::GeometricSplitting>>, partialACA>(argc, argv);
    test = test || test_hmat_zero<Cluster<PCA<SplittingTypes::GeometricSplitting>>, sympartialACA>(argc, argv);

    test = test || test_hmat_zero<Cluster<PCA<SplittingTypes::RegularSplitting>>, SVD>(argc, argv);

    test = test || test_hmat_zero<Cluster<PCA<SplittingTypes::RegularSplitting>>, fullACA>(argc, argv);
    test = test || test_hmat_zero<Cluster<PCA<SplittingTypes::RegularSplitting>>, partialACA>(argc, argv);
    test = test || test_hmat_zero<Cluster<PCA<SplittingTypes::RegularSplitting>>, sympartialACA>(argc, argv);

    // Finalize the MPI environment.
    MPI_Finalize();

    return test;
}
