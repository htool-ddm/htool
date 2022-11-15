#include "test_distributed_operator.hpp"

using namespace std;
using namespace htool;

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    bool test = 0;

    // Square matrix
    test = test || test_distributed_operator<std::complex<double>, GeneratorTestComplexHermitian, Cluster<PCA<SplittingTypes::GeometricSplitting>>>(200, 200, 5, true, 'N', 'N', 'T', false);
    test = test || test_distributed_operator<std::complex<double>, GeneratorTestComplex, Cluster<PCA<SplittingTypes::GeometricSplitting>>>(200, 200, 5, false, 'N', 'N', 'T', false);
    test = test || test_distributed_operator<std::complex<double>, GeneratorTestComplexHermitian, Cluster<PCA<SplittingTypes::GeometricSplitting>>>(200, 200, 5, true, 'H', 'L', 'T', false);
    test = test || test_distributed_operator<std::complex<double>, GeneratorTestComplexHermitian, Cluster<PCA<SplittingTypes::GeometricSplitting>>>(200, 200, 5, false, 'H', 'L', 'T', false);
    test = test || test_distributed_operator<std::complex<double>, GeneratorTestComplexHermitian, Cluster<PCA<SplittingTypes::GeometricSplitting>>>(200, 200, 5, true, 'H', 'U', 'T', false);
    test = test || test_distributed_operator<std::complex<double>, GeneratorTestComplexHermitian, Cluster<PCA<SplittingTypes::GeometricSplitting>>>(200, 200, 5, false, 'H', 'U', 'T', false);
    test = test || test_distributed_operator<std::complex<double>, GeneratorTestComplexSymmetric, Cluster<PCA<SplittingTypes::GeometricSplitting>>>(20, 20, 5, true, 'S', 'L', 'T', false);
    test = test || test_distributed_operator<std::complex<double>, GeneratorTestComplexSymmetric, Cluster<PCA<SplittingTypes::GeometricSplitting>>>(200, 200, 5, false, 'S', 'L', 'T', false);
    test = test || test_distributed_operator<std::complex<double>, GeneratorTestComplexSymmetric, Cluster<PCA<SplittingTypes::GeometricSplitting>>>(200, 200, 5, true, 'S', 'U', 'T', false);
    test = test || test_distributed_operator<std::complex<double>, GeneratorTestComplexSymmetric, Cluster<PCA<SplittingTypes::GeometricSplitting>>>(200, 200, 5, false, 'S', 'U', 'T', false);

    test = test || test_distributed_operator<std::complex<double>, GeneratorTestComplex, Cluster<PCA<SplittingTypes::GeometricSplitting>>>(200, 200, 5, true, 'N', 'N', 'T', true);
    test = test || test_distributed_operator<std::complex<double>, GeneratorTestComplex, Cluster<PCA<SplittingTypes::GeometricSplitting>>>(200, 200, 5, false, 'N', 'N', 'T', true);
    test = test || test_distributed_operator<std::complex<double>, GeneratorTestComplexHermitian, Cluster<PCA<SplittingTypes::GeometricSplitting>>>(200, 200, 5, true, 'H', 'L', 'T', true);
    test = test || test_distributed_operator<std::complex<double>, GeneratorTestComplexHermitian, Cluster<PCA<SplittingTypes::GeometricSplitting>>>(200, 200, 5, false, 'H', 'L', 'T', true);
    test = test || test_distributed_operator<std::complex<double>, GeneratorTestComplexHermitian, Cluster<PCA<SplittingTypes::GeometricSplitting>>>(200, 200, 5, true, 'H', 'U', 'T', true);
    test = test || test_distributed_operator<std::complex<double>, GeneratorTestComplexHermitian, Cluster<PCA<SplittingTypes::GeometricSplitting>>>(200, 200, 5, false, 'H', 'U', 'T', true);
    test = test || test_distributed_operator<std::complex<double>, GeneratorTestComplexSymmetric, Cluster<PCA<SplittingTypes::GeometricSplitting>>>(200, 200, 5, true, 'S', 'L', 'T', true);
    test = test || test_distributed_operator<std::complex<double>, GeneratorTestComplexSymmetric, Cluster<PCA<SplittingTypes::GeometricSplitting>>>(200, 200, 5, false, 'S', 'L', 'T', true);
    test = test || test_distributed_operator<std::complex<double>, GeneratorTestComplexSymmetric, Cluster<PCA<SplittingTypes::GeometricSplitting>>>(200, 200, 5, true, 'S', 'U', 'T', true);
    test = test || test_distributed_operator<std::complex<double>, GeneratorTestComplexSymmetric, Cluster<PCA<SplittingTypes::GeometricSplitting>>>(200, 200, 5, false, 'S', 'U', 'T', true);

    // Rectangular matrix
    test = test || test_distributed_operator<std::complex<double>, GeneratorTestComplex, Cluster<PCA<SplittingTypes::GeometricSplitting>>>(400, 200, 5, true, 'N', 'N', 'T', false);
    test = test || test_distributed_operator<std::complex<double>, GeneratorTestComplex, Cluster<PCA<SplittingTypes::GeometricSplitting>>>(400, 200, 5, false, 'N', 'N', 'T', false);
    test = test || test_distributed_operator<std::complex<double>, GeneratorTestComplex, Cluster<PCA<SplittingTypes::GeometricSplitting>>>(400, 200, 5, true, 'N', 'N', 'T', true);
    test = test || test_distributed_operator<std::complex<double>, GeneratorTestComplex, Cluster<PCA<SplittingTypes::GeometricSplitting>>>(400, 200, 5, false, 'N', 'N', 'T', true);

    test = test || test_distributed_operator<std::complex<double>, GeneratorTestComplex, Cluster<PCA<SplittingTypes::GeometricSplitting>>>(200, 400, 5, true, 'N', 'N', 'T', false);
    test = test || test_distributed_operator<std::complex<double>, GeneratorTestComplex, Cluster<PCA<SplittingTypes::GeometricSplitting>>>(200, 400, 5, false, 'N', 'N', 'T', false);
    test = test || test_distributed_operator<std::complex<double>, GeneratorTestComplex, Cluster<PCA<SplittingTypes::GeometricSplitting>>>(200, 400, 5, true, 'N', 'N', 'T', true);
    test = test || test_distributed_operator<std::complex<double>, GeneratorTestComplex, Cluster<PCA<SplittingTypes::GeometricSplitting>>>(200, 400, 5, false, 'N', 'N', 'T', true);
    MPI_Finalize();
    return test;
}
