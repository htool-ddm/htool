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

    bool test = test_virtual_hmat_product<std::complex<double>, GeneratorTestComplex, Cluster<PCA<SplittingTypes::GeometricSplitting>>, fullACA, HMatrix>(200, 100, 5, true, 'N', 'N', 'T', false);

    test = test || test_virtual_hmat_product<std::complex<double>, GeneratorTestComplex, Cluster<PCA<SplittingTypes::GeometricSplitting>>, fullACA, HMatrix>(200, 100, 5, false, 'N', 'N', 'T', false);

    test = test || test_virtual_hmat_product<std::complex<double>, GeneratorTestComplexHermitian, Cluster<PCA<SplittingTypes::GeometricSplitting>>, fullACA, HMatrix>(200, 200, 5, true, 'H', 'L', 'T', false);

    test = test || test_virtual_hmat_product<std::complex<double>, GeneratorTestComplexHermitian, Cluster<PCA<SplittingTypes::GeometricSplitting>>, fullACA, HMatrix>(200, 200, 5, false, 'H', 'L', 'T', false);

    test = test || test_virtual_hmat_product<std::complex<double>, GeneratorTestComplexHermitian, Cluster<PCA<SplittingTypes::GeometricSplitting>>, fullACA, HMatrix>(200, 200, 5, true, 'H', 'U', 'T', false);

    test = test || test_virtual_hmat_product<std::complex<double>, GeneratorTestComplexHermitian, Cluster<PCA<SplittingTypes::GeometricSplitting>>, fullACA, HMatrix>(200, 200, 5, false, 'H', 'U', 'T', false);

    test = test || test_virtual_hmat_product<std::complex<double>, GeneratorTestComplexSymmetric, Cluster<PCA<SplittingTypes::GeometricSplitting>>, fullACA, HMatrix>(200, 200, 5, true, 'S', 'L', 'T', false);

    test = test || test_virtual_hmat_product<std::complex<double>, GeneratorTestComplexSymmetric, Cluster<PCA<SplittingTypes::GeometricSplitting>>, fullACA, HMatrix>(200, 200, 5, false, 'S', 'L', 'T', false);

    test = test || test_virtual_hmat_product<std::complex<double>, GeneratorTestComplexSymmetric, Cluster<PCA<SplittingTypes::GeometricSplitting>>, fullACA, HMatrix>(200, 200, 5, true, 'S', 'U', 'T', false);

    test = test || test_virtual_hmat_product<std::complex<double>, GeneratorTestComplexSymmetric, Cluster<PCA<SplittingTypes::GeometricSplitting>>, fullACA, HMatrix>(200, 200, 5, false, 'S', 'U', 'T', false);

    // Finalize the MPI environment.
    MPI_Finalize();

    return test;
}
