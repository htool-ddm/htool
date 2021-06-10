#include "hmat.hpp"

int main(int argc, char *argv[]) {

    hmat<Cluster<PCARegularClustering>, partialACA>(argc, argv);

    return 0;
}
