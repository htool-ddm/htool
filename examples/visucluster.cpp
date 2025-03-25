#include <htool/clustering/implementations/partitioning.hpp>
#include <htool/htool.hpp>
#include <htool/testing/geometry.hpp>

using namespace std;
using namespace htool;

int main(int argc, char *argv[]) {

    // Check the number of parameters
    if (argc < 1) {
        // Tell the user how to run the program
        cerr << "Usage: " << argv[0] << "  outputname" << endl;
        /* "Usage messages" are a conventional way of telling the user
         * how to run a program if they enter the command incorrectly.
         */
        return 1;
    }
    std::string outputname = argv[1];

    // Geometry
    const int size              = 1000;
    const int spatial_dimension = 3;
    vector<double> p(spatial_dimension * size);
    create_sphere(size, p.data());

    // Clustering
    ClusterTreeBuilder<double> recursive_build_strategy;
    Cluster<double> cluster = recursive_build_strategy.create_cluster_tree(size, spatial_dimension, p.data(), 2, 2);

    // Output
    save_clustered_geometry(cluster, spatial_dimension, p.data(), outputname + "/clustering_output", {1, 2, 3});

    return 0;
}
