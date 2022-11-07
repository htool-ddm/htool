#include "test_distributed_operator.hpp"

using namespace std;
using namespace htool;

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    bool test = 0;

    // Square matrix
    test = test || test_distributed_operator<double, GeneratorTestDouble, Cluster<PCA<SplittingTypes::GeometricSplitting>>>(200, 200, 5, true, 'N', 'N', 'N', false);

    test = test || test_distributed_operator<double, GeneratorTestDouble, Cluster<PCA<SplittingTypes::GeometricSplitting>>>(200, 200, 5, false, 'N', 'N', 'N', false);

    test = test || test_distributed_operator<double, GeneratorTestDoubleSymmetric, Cluster<PCA<SplittingTypes::GeometricSplitting>>>(200, 200, 5, true, 'S', 'L', 'N', false);

    test = test || test_distributed_operator<double, GeneratorTestDoubleSymmetric, Cluster<PCA<SplittingTypes::GeometricSplitting>>>(200, 200, 5, false, 'S', 'L', 'N', false);

    test = test || test_distributed_operator<double, GeneratorTestDoubleSymmetric, Cluster<PCA<SplittingTypes::GeometricSplitting>>>(200, 200, 5, true, 'S', 'U', 'N', false);

    test = test || test_distributed_operator<double, GeneratorTestDoubleSymmetric, Cluster<PCA<SplittingTypes::GeometricSplitting>>>(200, 200, 5, false, 'S', 'U', 'N', false);

    test = test || test_distributed_operator<double, GeneratorTestDouble, Cluster<PCA<SplittingTypes::GeometricSplitting>>>(200, 200, 5, true, 'N', 'N', 'N', true);

    test = test || test_distributed_operator<double, GeneratorTestDouble, Cluster<PCA<SplittingTypes::GeometricSplitting>>>(200, 200, 5, false, 'N', 'N', 'N', true);

    test = test || test_distributed_operator<double, GeneratorTestDoubleSymmetric, Cluster<PCA<SplittingTypes::GeometricSplitting>>>(200, 200, 5, true, 'S', 'L', 'N', true);

    test = test || test_distributed_operator<double, GeneratorTestDoubleSymmetric, Cluster<PCA<SplittingTypes::GeometricSplitting>>>(200, 200, 5, false, 'S', 'L', 'N', true);

    test = test || test_distributed_operator<double, GeneratorTestDoubleSymmetric, Cluster<PCA<SplittingTypes::GeometricSplitting>>>(200, 200, 5, true, 'S', 'U', 'N', true);

    test = test || test_distributed_operator<double, GeneratorTestDoubleSymmetric, Cluster<PCA<SplittingTypes::GeometricSplitting>>>(200, 200, 5, false, 'S', 'U', 'N', true);

    // Rectangular matrix
    test = test || test_distributed_operator<double, GeneratorTestDouble, Cluster<PCA<SplittingTypes::GeometricSplitting>>>(400, 200, 5, true, 'N', 'N', 'N', false);
    test = test || test_distributed_operator<double, GeneratorTestDouble, Cluster<PCA<SplittingTypes::GeometricSplitting>>>(400, 200, 5, false, 'N', 'N', 'N', false);
    test = test || test_distributed_operator<double, GeneratorTestDouble, Cluster<PCA<SplittingTypes::GeometricSplitting>>>(400, 200, 5, true, 'N', 'N', 'N', true);
    test = test || test_distributed_operator<double, GeneratorTestDouble, Cluster<PCA<SplittingTypes::GeometricSplitting>>>(400, 200, 5, false, 'N', 'N', 'N', true);

    test = test || test_distributed_operator<double, GeneratorTestDouble, Cluster<PCA<SplittingTypes::GeometricSplitting>>>(200, 400, 5, true, 'N', 'N', 'N', false);
    test = test || test_distributed_operator<double, GeneratorTestDouble, Cluster<PCA<SplittingTypes::GeometricSplitting>>>(200, 400, 5, false, 'N', 'N', 'N', false);
    test = test || test_distributed_operator<double, GeneratorTestDouble, Cluster<PCA<SplittingTypes::GeometricSplitting>>>(200, 400, 5, true, 'N', 'N', 'N', true);
    test = test || test_distributed_operator<double, GeneratorTestDouble, Cluster<PCA<SplittingTypes::GeometricSplitting>>>(200, 400, 5, false, 'N', 'N', 'N', true);
    MPI_Finalize();
    return test;
}
