#ifndef HTOOL_HMATRIX_BUILDER_HPP
#define HTOOL_HMATRIX_BUILDER_HPP

#include "../clustering/cluster_node.hpp"
#include "../clustering/tree_builder/tree_builder.hpp"
#include "hmatrix.hpp"
#include "tree_builder/tree_builder.hpp"

namespace htool {

template <typename CoefficientsPrecision, typename CoordinatesPrecision = htool::underlying_type<CoefficientsPrecision>>
class HMatrixBuilder {
  public:
    Cluster<CoordinatesPrecision> target_cluster;
    std::unique_ptr<Cluster<CoordinatesPrecision>> source_cluster_ptr = nullptr; // use an optional source cluster

    HMatrixBuilder(int target_number_of_points, int target_spatial_dimension, const CoordinatesPrecision *target_coordinates, const ClusterTreeBuilder<CoordinatesPrecision> *target_cluster_tree_builder, int source_number_of_points, int source_spatial_dimension, const CoordinatesPrecision *source_coordinates, const ClusterTreeBuilder<CoordinatesPrecision> *source_cluster_tree_builder) : target_cluster(target_cluster_tree_builder == nullptr ? ClusterTreeBuilder<CoordinatesPrecision>().create_cluster_tree(target_number_of_points, target_spatial_dimension, target_coordinates, std::pow(2, target_spatial_dimension), std::pow(2, source_spatial_dimension)) : target_cluster_tree_builder->create_cluster_tree(target_number_of_points, target_spatial_dimension, target_coordinates, 2, 2)), source_cluster_ptr(std::make_unique<Cluster<CoordinatesPrecision>>(source_cluster_tree_builder == nullptr ? ClusterTreeBuilder<CoordinatesPrecision>().create_cluster_tree(source_number_of_points, source_spatial_dimension, source_coordinates, std::pow(2, target_spatial_dimension), std::pow(2, source_spatial_dimension)) : source_cluster_tree_builder->create_cluster_tree(source_number_of_points, source_spatial_dimension, source_coordinates, std::pow(2, target_spatial_dimension), std::pow(2, source_spatial_dimension)))) {
    }

    HMatrixBuilder(int target_number_of_points, int target_spatial_dimension, const CoordinatesPrecision *target_coordinates, int source_number_of_points, int source_spatial_dimension, const CoordinatesPrecision *source_coordinates) : HMatrixBuilder(target_number_of_points, target_spatial_dimension, target_coordinates, nullptr, source_number_of_points, source_spatial_dimension, source_coordinates, nullptr) {
    }

    HMatrixBuilder(int number_of_points, int spatial_dimension, const CoordinatesPrecision *coordinates, const ClusterTreeBuilder<CoordinatesPrecision> *cluster_tree_builder = nullptr) : target_cluster(cluster_tree_builder == nullptr ? ClusterTreeBuilder<CoordinatesPrecision>().create_cluster_tree(number_of_points, spatial_dimension, coordinates, std::pow(2, spatial_dimension), std::pow(2, spatial_dimension)) : cluster_tree_builder->create_cluster_tree(number_of_points, spatial_dimension, coordinates, std::pow(2, spatial_dimension), std::pow(2, spatial_dimension))) {
    }

    HMatrix<CoefficientsPrecision, CoordinatesPrecision> build(const VirtualGenerator<CoefficientsPrecision> &generator, const HMatrixTreeBuilder<CoefficientsPrecision, CoordinatesPrecision> &hmatrix_tree_builder) {
        return hmatrix_tree_builder.build(generator, target_cluster, source_cluster_ptr == nullptr ? target_cluster : *source_cluster_ptr);
    }
};
} // namespace htool
#endif
