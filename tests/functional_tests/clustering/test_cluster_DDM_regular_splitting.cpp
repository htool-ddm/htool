#include "test_cluster.hpp"
#include "test_cluster_DDM.hpp"

using namespace std;
using namespace htool;


int main(int argc, char *argv[]) {
    MPI_Init(&argc,&argv);
    
    std::cout << "Automatic partition"<< std::endl;
    bool test = test_cluster<RegularClusteringDDM>(argc,argv);

    std::cout << "Given partition"<< std::endl;
    test = test || (test_cluster_DDM<SplittingTypes::RegularSplitting>(argc,argv));
    MPI_Finalize();
    return test;

}