#include <htool/clustering/cluster_output.hpp>
#include <htool/clustering/tree_builder/recursive_build.hpp>
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
    int size = 1000;
    vector<double> p(3 * size);
    create_sphere(size, p.data());

    // Clustering
    ClusterTreeBuilder<double> recursive_build_strategy;
    Cluster<double> cluster = recursive_build_strategy.create_cluster_tree(size, 3, p.data(), 2, 2);

    // Output
    save_clustered_geometry(cluster, 3, p.data(), outputname + "/clustering_output", {1, 2, 3});

    return 0;
}
