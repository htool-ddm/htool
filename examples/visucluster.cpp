#include <htool/clustering/pca.hpp>

using namespace std;
using namespace htool;

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

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
    double z = 1;
    vector<double> p(3 * size);

    for (int j = 0; j < size; j++) {
        double rho   = ((double)rand() / (double)(RAND_MAX)); // (double) otherwise integer division!
        double theta = ((double)rand() / (double)(RAND_MAX));
        p[j + 0]     = sqrt(rho) * cos(2 * M_PI * theta);
        p[j + 1]     = sqrt(rho) * sin(2 * M_PI * theta);
        p[j + 2]     = z;
        // sqrt(rho) otherwise the points would be concentrated in the center of the disk
    }

    // Clustering
    Cluster<PCA<SplittingTypes::GeometricSplitting>> t(3);
    t.build(size, p.data(), 2);

    // Output
    t.save_geometry(p.data(), outputname + "/clustering_output", {1, 2, 3});

    std::cout << outputname + "/clustering_output" << std::endl;
    MPI_Finalize();
    return 0;
}
