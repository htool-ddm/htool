#include <htool/clustering/ncluster.hpp>
#include "test_cluster.hpp"

using namespace std;
using namespace htool;


int main(int argc, char *argv[]) {
    
    MPI_Init(&argc,&argv);
    
    bool test = test_cluster<GeometricClustering>(argc,argv);

    MPI_Finalize();
    return test;

}