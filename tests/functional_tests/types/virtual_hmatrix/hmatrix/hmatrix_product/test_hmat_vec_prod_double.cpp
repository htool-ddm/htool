#include "../../test_virtual_hmat_product.hpp"
#include <htool/clustering/pca.hpp>
#include <htool/lrmat/fullACA.hpp>
#include <htool/lrmat/partialACA.hpp>
#include <htool/testing/generator_test.hpp>
#include <htool/testing/geometry.hpp>
#include <htool/types/hmatrix.hpp>

using namespace std;
using namespace htool;

int main(int argc, char *argv[]) {

    // Initialize the MPI environment
    MPI_Init(&argc, &argv);

    bool test = test_virtual_hmat_product<double, GeneratorTestDouble, Cluster<PCA<SplittingTypes::GeometricSplitting>>, fullACA, HMatrix>(2000, 1000, 1, true, 'N', 'N', 'N', false);

    test = test || test_virtual_hmat_product<double, GeneratorTestDouble, Cluster<PCA<SplittingTypes::GeometricSplitting>>, fullACA, HMatrix>(2000, 1000, 1, false, 'N', 'N', 'N', false);

    test = test || test_virtual_hmat_product<double, GeneratorTestDoubleSymmetric, Cluster<PCA<SplittingTypes::GeometricSplitting>>, fullACA, HMatrix>(2000, 2000, 1, true, 'S', 'L', 'N', false);

    test = test || test_virtual_hmat_product<double, GeneratorTestDoubleSymmetric, Cluster<PCA<SplittingTypes::GeometricSplitting>>, fullACA, HMatrix>(2000, 2000, 1, false, 'S', 'L', 'N', false);

    test = test || test_virtual_hmat_product<double, GeneratorTestDoubleSymmetric, Cluster<PCA<SplittingTypes::GeometricSplitting>>, fullACA, HMatrix>(2000, 2000, 1, true, 'S', 'U', 'N', false);

    test = test || test_virtual_hmat_product<double, GeneratorTestDoubleSymmetric, Cluster<PCA<SplittingTypes::GeometricSplitting>>, partialACA, HMatrix>(2000, 2000, 1, false, 'S', 'U', 'N', false);

    test = test || test_virtual_hmat_product<double, GeneratorTestDouble, Cluster<PCA<SplittingTypes::GeometricSplitting>>, fullACA, HMatrix>(2000, 1000, 1, true, 'N', 'N', 'N', true);

    test = test || test_virtual_hmat_product<double, GeneratorTestDouble, Cluster<PCA<SplittingTypes::GeometricSplitting>>, fullACA, HMatrix>(2000, 1000, 1, false, 'N', 'N', 'N', true);

    test = test || test_virtual_hmat_product<double, GeneratorTestDoubleSymmetric, Cluster<PCA<SplittingTypes::GeometricSplitting>>, fullACA, HMatrix>(2000, 2000, 1, true, 'S', 'L', 'N', true);

    test = test || test_virtual_hmat_product<double, GeneratorTestDoubleSymmetric, Cluster<PCA<SplittingTypes::GeometricSplitting>>, fullACA, HMatrix>(2000, 2000, 1, false, 'S', 'L', 'N', true);

    test = test || test_virtual_hmat_product<double, GeneratorTestDoubleSymmetric, Cluster<PCA<SplittingTypes::GeometricSplitting>>, fullACA, HMatrix>(2000, 2000, 1, true, 'S', 'U', 'N', true);

    test = test || test_virtual_hmat_product<double, GeneratorTestDoubleSymmetric, Cluster<PCA<SplittingTypes::GeometricSplitting>>, partialACA, HMatrix>(2000, 2000, 1, false, 'S', 'U', 'N', true);

    // Finalize the MPI environment.
    MPI_Finalize();

    return test;
}
