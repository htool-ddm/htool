#include <htool/clustering/cluster_output.hpp>
#include <htool/clustering/implementations/partitioning.hpp>
#include <htool/clustering/tree_builder/tree_builder.hpp>
#include <htool/testing/geometry.hpp>

using namespace std;
using namespace htool;

int main(int argc, char *argv[]) {

    // Check the number of parameters
    if (argc > 2) {
        // Tell the user how to run the program
        cerr << "Usage: " << argv[0] << "  output_folder" << endl;
        /* "Usage messages" are a conventional way of telling the user
         * how to run a program if they enter the command incorrectly.
         */
        return 1;
    }
    std::string output_folder = argc == 2 ? argv[1] : "./";

    // Geometry
    const int number_points        = 10000;
    const int spatial_dimension    = 3;
    const int number_of_partitions = 8;
    const int number_of_children   = 8;
    vector<double> coordinates(spatial_dimension * number_points);
    create_sphere(number_points, coordinates.data());

    // Cluster tree builder with customization
    ClusterTreeBuilder<double> recursive_build_strategy;
    recursive_build_strategy.set_maximal_leaf_size(10);
    recursive_build_strategy.set_partitioning_strategy(std::make_shared<Partitioning_N<double, ComputeLargestExtent<double>, RegularSplitting<double>>>());

    // Clustering
    Cluster<double> cluster = recursive_build_strategy.create_cluster_tree(number_points, spatial_dimension, coordinates.data(), number_of_children, number_of_partitions);

    // Output
    save_clustered_geometry(cluster, spatial_dimension, coordinates.data(), output_folder + "/clustering_output", {1, 2, 3});

    return 0;
}
