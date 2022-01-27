#include "../../test_virtual_hmat_product.hpp"
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

    bool test = test_virtual_hmat_product<double, GeneratorTestDouble, Cluster<PCA<SplittingTypes::GeometricSplitting>>, fullACA, HMatrix>(2000, 1000, 1, true, 'N', 'N', 'T');

    test = test || test_virtual_hmat_product<double, GeneratorTestDouble, Cluster<PCA<SplittingTypes::GeometricSplitting>>, fullACA, HMatrix>(2000, 1000, 1, false, 'N', 'N', 'T');

    test = test || test_virtual_hmat_product<double, GeneratorTestDoubleSymmetric, Cluster<PCA<SplittingTypes::GeometricSplitting>>, fullACA, HMatrix>(2000, 2000, 1, true, 'S', 'L', 'T');

    test = test || test_virtual_hmat_product<double, GeneratorTestDoubleSymmetric, Cluster<PCA<SplittingTypes::GeometricSplitting>>, fullACA, HMatrix>(2000, 2000, 1, false, 'S', 'L', 'T');

    test = test || test_virtual_hmat_product<double, GeneratorTestDoubleSymmetric, Cluster<PCA<SplittingTypes::GeometricSplitting>>, fullACA, HMatrix>(2000, 2000, 1, true, 'S', 'U', 'T');

    test = test || test_virtual_hmat_product<double, GeneratorTestDoubleSymmetric, Cluster<PCA<SplittingTypes::GeometricSplitting>>, fullACA, HMatrix>(2000, 2000, 1, false, 'S', 'U', 'T');

    // Finalize the MPI environment.
    MPI_Finalize();

    return test;
}
