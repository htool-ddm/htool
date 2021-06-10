#include "hmat.hpp"

int main(int argc, char *argv[]) {

    hmat<Cluster<PCARegularClustering>, fullACA>(argc, argv);

    return 0;
}
