#include "test_cluster.hpp"
#include <cmath> // for pow
#include <htool/clustering/implementations/partitioning.hpp>

using namespace std;
using namespace htool;

int main(int argc, char *argv[]) {

    MPI_Init(&argc, &argv);

    bool is_error  = false;
    const int size = 200;

    for (int dimension : {2, 3}) {
        for (auto partition_type : {PartitionType::None, PartitionType::Local, PartitionType::Global}) {
            is_error = is_error || test_cluster<double, htool::Partitioning<double, htool::ComputeLargestExtent<double>, htool::RegularSplitting<double>>>(dimension, size, partition_type);
            is_error = is_error || test_cluster<double, htool::Partitioning<double, htool::ComputeLargestExtent<double>, htool::GeometricSplitting<double>>>(dimension, size, partition_type);
            is_error = is_error || test_cluster<double, htool::Partitioning<double, htool::ComputeBoundingBox<double>, htool::RegularSplitting<double>>>(dimension, size, partition_type);
            is_error = is_error || test_cluster<double, htool::Partitioning<double, htool::ComputeBoundingBox<double>, htool::GeometricSplitting<double>>>(dimension, size, partition_type);

            is_error = is_error || test_cluster<float, htool::Partitioning<float, htool::ComputeLargestExtent<float>, htool::RegularSplitting<float>>>(dimension, size, partition_type);
            is_error = is_error || test_cluster<float, htool::Partitioning<float, htool::ComputeLargestExtent<float>, htool::GeometricSplitting<float>>>(dimension, size, partition_type);
            is_error = is_error || test_cluster<float, htool::Partitioning<float, htool::ComputeBoundingBox<float>, htool::RegularSplitting<float>>>(dimension, size, partition_type);
            is_error = is_error || test_cluster<float, htool::Partitioning<float, htool::ComputeBoundingBox<float>, htool::GeometricSplitting<float>>>(dimension, size, partition_type);

            is_error = is_error || test_cluster<double, htool::Partitioning_N<double, htool::ComputeLargestExtent<double>, htool::RegularSplitting<double>>>(dimension, size, partition_type);
            is_error = is_error || test_cluster<double, htool::Partitioning_N<double, htool::ComputeLargestExtent<double>, htool::GeometricSplitting<double>>>(dimension, size, partition_type);
            is_error = is_error || test_cluster<double, htool::Partitioning_N<double, htool::ComputeBoundingBox<double>, htool::RegularSplitting<double>>>(dimension, size, partition_type);
            is_error = is_error || test_cluster<double, htool::Partitioning_N<double, htool::ComputeBoundingBox<double>, htool::GeometricSplitting<double>>>(dimension, size, partition_type);

            is_error = is_error || test_cluster<float, htool::Partitioning_N<float, htool::ComputeLargestExtent<float>, htool::RegularSplitting<float>>>(dimension, size, partition_type);
            is_error = is_error || test_cluster<float, htool::Partitioning_N<float, htool::ComputeLargestExtent<float>, htool::GeometricSplitting<float>>>(dimension, size, partition_type);
            is_error = is_error || test_cluster<float, htool::Partitioning_N<float, htool::ComputeBoundingBox<float>, htool::RegularSplitting<float>>>(dimension, size, partition_type);
            is_error = is_error || test_cluster<float, htool::Partitioning_N<float, htool::ComputeBoundingBox<float>, htool::GeometricSplitting<float>>>(dimension, size, partition_type);
        }
    }

    MPI_Finalize();
    if (is_error) {
        return 1;
    }
    return 0;
}
