#include "test_distributed_operator.hpp"

using namespace std;
using namespace htool;

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    bool test = 0;

    // Square matrix
    test = test || test_distributed_operator<double, GeneratorTestDouble, Cluster<PCA<SplittingTypes::GeometricSplitting>>>(200, 200, 1, true, 'N', 'N', 'T', false);
    test = test || test_distributed_operator<double, GeneratorTestDouble, Cluster<PCA<SplittingTypes::GeometricSplitting>>>(200, 200, 1, false, 'N', 'N', 'T', false);
    test = test || test_distributed_operator<double, GeneratorTestDoubleSymmetric, Cluster<PCA<SplittingTypes::GeometricSplitting>>>(200, 200, 1, true, 'S', 'L', 'T', false);
    test = test || test_distributed_operator<double, GeneratorTestDoubleSymmetric, Cluster<PCA<SplittingTypes::GeometricSplitting>>>(200, 200, 1, false, 'S', 'L', 'T', false);
    test = test || test_distributed_operator<double, GeneratorTestDoubleSymmetric, Cluster<PCA<SplittingTypes::GeometricSplitting>>>(200, 200, 1, true, 'S', 'U', 'T', false);
    test = test || test_distributed_operator<double, GeneratorTestDoubleSymmetric, Cluster<PCA<SplittingTypes::GeometricSplitting>>>(200, 200, 1, false, 'S', 'U', 'T', false);

    test = test || test_distributed_operator<double, GeneratorTestDouble, Cluster<PCA<SplittingTypes::GeometricSplitting>>>(200, 200, 1, true, 'N', 'N', 'T', true);
    test = test || test_distributed_operator<double, GeneratorTestDouble, Cluster<PCA<SplittingTypes::GeometricSplitting>>>(200, 200, 1, false, 'N', 'N', 'T', true);
    test = test || test_distributed_operator<double, GeneratorTestDoubleSymmetric, Cluster<PCA<SplittingTypes::GeometricSplitting>>>(200, 200, 1, true, 'S', 'L', 'T', true);
    test = test || test_distributed_operator<double, GeneratorTestDoubleSymmetric, Cluster<PCA<SplittingTypes::GeometricSplitting>>>(200, 200, 1, false, 'S', 'L', 'T', true);
    test = test || test_distributed_operator<double, GeneratorTestDoubleSymmetric, Cluster<PCA<SplittingTypes::GeometricSplitting>>>(200, 200, 1, true, 'S', 'U', 'T', true);
    test = test || test_distributed_operator<double, GeneratorTestDoubleSymmetric, Cluster<PCA<SplittingTypes::GeometricSplitting>>>(200, 200, 1, false, 'S', 'U', 'T', true);

    // Rectangular matrix
    test = test || test_distributed_operator<double, GeneratorTestDouble, Cluster<PCA<SplittingTypes::GeometricSplitting>>>(400, 200, 1, true, 'N', 'N', 'T', false);
    test = test || test_distributed_operator<double, GeneratorTestDouble, Cluster<PCA<SplittingTypes::GeometricSplitting>>>(400, 200, 1, false, 'N', 'N', 'T', false);
    test = test || test_distributed_operator<double, GeneratorTestDouble, Cluster<PCA<SplittingTypes::GeometricSplitting>>>(400, 200, 1, true, 'N', 'N', 'T', true);
    test = test || test_distributed_operator<double, GeneratorTestDouble, Cluster<PCA<SplittingTypes::GeometricSplitting>>>(400, 200, 1, false, 'N', 'N', 'T', true);

    test = test || test_distributed_operator<double, GeneratorTestDouble, Cluster<PCA<SplittingTypes::GeometricSplitting>>>(200, 400, 1, true, 'N', 'N', 'T', false);
    test = test || test_distributed_operator<double, GeneratorTestDouble, Cluster<PCA<SplittingTypes::GeometricSplitting>>>(200, 400, 1, false, 'N', 'N', 'T', false);
    test = test || test_distributed_operator<double, GeneratorTestDouble, Cluster<PCA<SplittingTypes::GeometricSplitting>>>(200, 400, 1, true, 'N', 'N', 'T', true);
    test = test || test_distributed_operator<double, GeneratorTestDouble, Cluster<PCA<SplittingTypes::GeometricSplitting>>>(200, 400, 1, false, 'N', 'N', 'T', true);
    MPI_Finalize();
    return test;
}
