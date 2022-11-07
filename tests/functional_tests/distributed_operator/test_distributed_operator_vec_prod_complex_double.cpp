#include "test_distributed_operator.hpp"

using namespace std;
using namespace htool;

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    bool test = 0;

    // Square matrix
    test = test || test_distributed_operator<std::complex<double>, GeneratorTestComplex, Cluster<PCA<SplittingTypes::GeometricSplitting>>>(200, 200, 1, true, 'N', 'N', 'N', false);

    test = test || test_distributed_operator<std::complex<double>, GeneratorTestComplex, Cluster<PCA<SplittingTypes::GeometricSplitting>>>(200, 200, 1, false, 'N', 'N', 'N', false);

    test = test || test_distributed_operator<std::complex<double>, GeneratorTestComplexHermitian, Cluster<PCA<SplittingTypes::GeometricSplitting>>>(200, 200, 1, true, 'H', 'L', 'N', false);

    test = test || test_distributed_operator<std::complex<double>, GeneratorTestComplexHermitian, Cluster<PCA<SplittingTypes::GeometricSplitting>>>(200, 200, 1, false, 'H', 'L', 'N', false);

    test = test || test_distributed_operator<std::complex<double>, GeneratorTestComplexHermitian, Cluster<PCA<SplittingTypes::GeometricSplitting>>>(200, 200, 1, true, 'H', 'U', 'N', false);

    test = test || test_distributed_operator<std::complex<double>, GeneratorTestComplexHermitian, Cluster<PCA<SplittingTypes::GeometricSplitting>>>(200, 200, 1, false, 'H', 'U', 'N', false);

    test = test || test_distributed_operator<std::complex<double>, GeneratorTestComplexSymmetric, Cluster<PCA<SplittingTypes::GeometricSplitting>>>(200, 200, 1, true, 'S', 'L', 'N', false);

    test = test || test_distributed_operator<std::complex<double>, GeneratorTestComplexSymmetric, Cluster<PCA<SplittingTypes::GeometricSplitting>>>(200, 200, 1, false, 'S', 'L', 'N', false);

    test = test || test_distributed_operator<std::complex<double>, GeneratorTestComplexSymmetric, Cluster<PCA<SplittingTypes::GeometricSplitting>>>(200, 200, 1, true, 'S', 'U', 'N', false);

    test = test || test_distributed_operator<std::complex<double>, GeneratorTestComplexSymmetric, Cluster<PCA<SplittingTypes::GeometricSplitting>>>(200, 200, 1, false, 'S', 'U', 'N', false);

    test = test || test_distributed_operator<std::complex<double>, GeneratorTestComplex, Cluster<PCA<SplittingTypes::GeometricSplitting>>>(200, 200, 1, true, 'N', 'N', 'N', true);

    test = test || test_distributed_operator<std::complex<double>, GeneratorTestComplex, Cluster<PCA<SplittingTypes::GeometricSplitting>>>(200, 200, 1, false, 'N', 'N', 'N', true);

    test = test || test_distributed_operator<std::complex<double>, GeneratorTestComplexHermitian, Cluster<PCA<SplittingTypes::GeometricSplitting>>>(200, 200, 1, true, 'H', 'L', 'N', true);

    test = test || test_distributed_operator<std::complex<double>, GeneratorTestComplexHermitian, Cluster<PCA<SplittingTypes::GeometricSplitting>>>(200, 200, 1, false, 'H', 'L', 'N', true);

    test = test || test_distributed_operator<std::complex<double>, GeneratorTestComplexHermitian, Cluster<PCA<SplittingTypes::GeometricSplitting>>>(200, 200, 1, true, 'H', 'U', 'N', true);

    test = test || test_distributed_operator<std::complex<double>, GeneratorTestComplexHermitian, Cluster<PCA<SplittingTypes::GeometricSplitting>>>(200, 200, 1, false, 'H', 'U', 'N', true);

    test = test || test_distributed_operator<std::complex<double>, GeneratorTestComplexSymmetric, Cluster<PCA<SplittingTypes::GeometricSplitting>>>(200, 200, 1, true, 'S', 'L', 'N', true);

    test = test || test_distributed_operator<std::complex<double>, GeneratorTestComplexSymmetric, Cluster<PCA<SplittingTypes::GeometricSplitting>>>(200, 200, 1, false, 'S', 'L', 'N', true);

    test = test || test_distributed_operator<std::complex<double>, GeneratorTestComplexSymmetric, Cluster<PCA<SplittingTypes::GeometricSplitting>>>(200, 200, 1, true, 'S', 'U', 'N', true);

    test = test || test_distributed_operator<std::complex<double>, GeneratorTestComplexSymmetric, Cluster<PCA<SplittingTypes::GeometricSplitting>>>(200, 200, 1, false, 'S', 'U', 'N', true);

    // Rectangular matrix
    test = test || test_distributed_operator<std::complex<double>, GeneratorTestComplex, Cluster<PCA<SplittingTypes::GeometricSplitting>>>(400, 200, 1, true, 'N', 'N', 'N', false);
    test = test || test_distributed_operator<std::complex<double>, GeneratorTestComplex, Cluster<PCA<SplittingTypes::GeometricSplitting>>>(400, 200, 1, false, 'N', 'N', 'N', false);
    test = test || test_distributed_operator<std::complex<double>, GeneratorTestComplex, Cluster<PCA<SplittingTypes::GeometricSplitting>>>(400, 200, 1, true, 'N', 'N', 'N', true);
    test = test || test_distributed_operator<std::complex<double>, GeneratorTestComplex, Cluster<PCA<SplittingTypes::GeometricSplitting>>>(400, 200, 1, false, 'N', 'N', 'N', true);

    test = test || test_distributed_operator<std::complex<double>, GeneratorTestComplex, Cluster<PCA<SplittingTypes::GeometricSplitting>>>(200, 400, 1, true, 'N', 'N', 'N', false);
    test = test || test_distributed_operator<std::complex<double>, GeneratorTestComplex, Cluster<PCA<SplittingTypes::GeometricSplitting>>>(200, 400, 1, false, 'N', 'N', 'N', false);
    test = test || test_distributed_operator<std::complex<double>, GeneratorTestComplex, Cluster<PCA<SplittingTypes::GeometricSplitting>>>(200, 400, 1, true, 'N', 'N', 'N', true);
    test = test || test_distributed_operator<std::complex<double>, GeneratorTestComplex, Cluster<PCA<SplittingTypes::GeometricSplitting>>>(200, 400, 1, false, 'N', 'N', 'N', true);
    MPI_Finalize();
    return test;
}
