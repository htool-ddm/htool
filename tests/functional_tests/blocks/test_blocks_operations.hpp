#include <htool/blocks/admissibility_conditions.hpp>
#include <htool/blocks/blocks.hpp>
#include <htool/blocks/blocks_operations.hpp>
#include <htool/clustering/pca.hpp>
#include <htool/lrmat/fullACA.hpp>
#include <htool/testing/generator_test.hpp>
#include <htool/testing/geometry.hpp>
#include <random>

using namespace std;
using namespace htool;

template <class AdmissibilityCondition>
int test_blocks_operations(int argc, char *argv[], bool symmetric) {

    int rankWorld, sizeWorld;
    MPI_Comm_size(MPI_COMM_WORLD, &sizeWorld);
    MPI_Comm_rank(MPI_COMM_WORLD, &rankWorld);

    srand(1);
    bool test      = 0;
    double epsilon = 1e-3;
    double eta     = 10;
    int size       = 200;
    double z       = 1;
    vector<double> p(3 * size);

    srand(1);
    // we set a constant seed for rand because we want always the same result if we run the check many times
    // (two different initializations with the same seed will generate the same succession of results in the subsequent calls to rand)
    create_disk(3, z, size, p.data());

    Cluster<PCAGeometricClustering> t;
    t.build(size, p.data());

    std::unique_ptr<AdmissibilityCondition> admissibility_condition(new AdmissibilityCondition());
    Block<double> B(admissibility_condition.get(), t, t);
    B.set_eta(eta);
    B.build(symmetric);

    const std::vector<Block<double> *> &local_tasks = B.get_local_tasks();

    // Fill blocks
    GeneratorTestDoubleSymmetric A(3, size, size, p, p);
    std::shared_ptr<VirtualLowRankGenerator<double>> compressor = std::make_shared<fullACA<double>>();
    for (int i = 0; i < local_tasks.size(); i++) {
        if (local_tasks[i]->IsAdmissible()) {
            local_tasks[i]->compute_low_rank_block(-1, epsilon, A, *compressor, p.data(), p.data(), true);
            if (local_tasks[i]->get_rank_of() == -1) {
                local_tasks[i]->clear_data();
                local_tasks[i]->compute_dense_block(A, true);
            }
        } else {
            local_tasks[i]->compute_dense_block(A, true);
        }
    }

    // Test scale
    std::vector<double> x(size, 1), y(size);
    Hmatvec(B, x.data(), y.data());
    test = test || !(max(y) > 1e-16);
    Hscale(B, 0.);
    Hmatvec(B, x.data(), y.data());
    test = test || !(max(y) < 1e-16);

    // Clear blocks
    for (int i = 0; i < local_tasks.size(); i++) {
        local_tasks[i]->clear_data();
    }

    // Fill blocks again
    for (int i = 0; i < local_tasks.size(); i++) {
        if (local_tasks[i]->IsAdmissible()) {
            local_tasks[i]->compute_low_rank_block(-1, epsilon, A, *compressor, p.data(), p.data(), true);
            if (local_tasks[i]->get_rank_of() == -1) {
                local_tasks[i]->clear_data();
                local_tasks[i]->compute_dense_block(A, true);
            }
        } else {
            local_tasks[i]->compute_dense_block(A, true);
        }
    }

    // double max = 0;
    // for (int i = 0; i < local_tasks.size(); i++) {
    //     max =
    // }

    if (rankWorld == 0) {
        std::cout << "test " << test << std::endl;
    }

    return test;
}
