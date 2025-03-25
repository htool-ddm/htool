#include "test_cluster.hpp"
#include <cmath> // for pow
#include <htool/clustering/implementations/partitioning.hpp>

using namespace std;
using namespace htool;

int main(int argc, char *argv[]) {

    MPI_Init(&argc, &argv);

    bool is_error  = false;
    const int size = 200;

    is_error = is_error || test_cluster<double, 2, htool::ComputeLargestExtent<double>, htool::RegularSplitting<double>>(size, false);
    is_error = is_error || test_cluster<double, 2, htool::ComputeLargestExtent<double>, htool::GeometricSplitting<double>>(size, false);
    is_error = is_error || test_cluster<double, 2, htool::ComputeBoundingBox<double>, htool::RegularSplitting<double>>(size, false);
    is_error = is_error || test_cluster<double, 2, htool::ComputeBoundingBox<double>, htool::GeometricSplitting<double>>(size, false);

    is_error = is_error || test_cluster<double, 2, htool::ComputeLargestExtent<double>, htool::RegularSplitting<double>>(size, true);
    is_error = is_error || test_cluster<double, 2, htool::ComputeLargestExtent<double>, htool::GeometricSplitting<double>>(size, true);
    is_error = is_error || test_cluster<double, 2, htool::ComputeBoundingBox<double>, htool::RegularSplitting<double>>(size, true);
    is_error = is_error || test_cluster<double, 2, htool::ComputeBoundingBox<double>, htool::GeometricSplitting<double>>(size, true);

    is_error = is_error || test_cluster<double, 3, htool::ComputeLargestExtent<double>, htool::RegularSplitting<double>>(size, false);
    is_error = is_error || test_cluster<double, 3, htool::ComputeLargestExtent<double>, htool::GeometricSplitting<double>>(size, false);
    is_error = is_error || test_cluster<double, 3, htool::ComputeBoundingBox<double>, htool::RegularSplitting<double>>(size, false);
    is_error = is_error || test_cluster<double, 3, htool::ComputeBoundingBox<double>, htool::GeometricSplitting<double>>(size, false);

    is_error = is_error || test_cluster<double, 3, htool::ComputeLargestExtent<double>, htool::RegularSplitting<double>>(size, true);
    is_error = is_error || test_cluster<double, 3, htool::ComputeLargestExtent<double>, htool::GeometricSplitting<double>>(size, true);
    is_error = is_error || test_cluster<double, 3, htool::ComputeBoundingBox<double>, htool::RegularSplitting<double>>(size, true);
    is_error = is_error || test_cluster<double, 3, htool::ComputeBoundingBox<double>, htool::GeometricSplitting<double>>(size, true);

    is_error = is_error || test_cluster<float, 2, htool::ComputeLargestExtent<float>, htool::RegularSplitting<float>>(size, false);
    is_error = is_error || test_cluster<float, 2, htool::ComputeLargestExtent<float>, htool::GeometricSplitting<float>>(size, false);
    is_error = is_error || test_cluster<float, 2, htool::ComputeBoundingBox<float>, htool::RegularSplitting<float>>(size, false);
    is_error = is_error || test_cluster<float, 2, htool::ComputeBoundingBox<float>, htool::GeometricSplitting<float>>(size, false);

    is_error = is_error || test_cluster<float, 2, htool::ComputeLargestExtent<float>, htool::RegularSplitting<float>>(size, true);
    is_error = is_error || test_cluster<float, 2, htool::ComputeLargestExtent<float>, htool::GeometricSplitting<float>>(size, true);
    is_error = is_error || test_cluster<float, 2, htool::ComputeBoundingBox<float>, htool::RegularSplitting<float>>(size, true);
    is_error = is_error || test_cluster<float, 2, htool::ComputeBoundingBox<float>, htool::GeometricSplitting<float>>(size, true);

    is_error = is_error || test_cluster<float, 3, htool::ComputeLargestExtent<float>, htool::RegularSplitting<float>>(size, false);
    is_error = is_error || test_cluster<float, 3, htool::ComputeLargestExtent<float>, htool::GeometricSplitting<float>>(size, false);
    is_error = is_error || test_cluster<float, 3, htool::ComputeBoundingBox<float>, htool::RegularSplitting<float>>(size, false);
    is_error = is_error || test_cluster<float, 3, htool::ComputeBoundingBox<float>, htool::GeometricSplitting<float>>(size, false);

    is_error = is_error || test_cluster<float, 3, htool::ComputeLargestExtent<float>, htool::RegularSplitting<float>>(size, true);
    is_error = is_error || test_cluster<float, 3, htool::ComputeLargestExtent<float>, htool::GeometricSplitting<float>>(size, true);
    is_error = is_error || test_cluster<float, 3, htool::ComputeBoundingBox<float>, htool::RegularSplitting<float>>(size, true);
    is_error = is_error || test_cluster<float, 3, htool::ComputeBoundingBox<float>, htool::GeometricSplitting<float>>(size, true);

    MPI_Finalize();
    if (is_error) {
        return 1;
    }
    return 0;
}
