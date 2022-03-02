#include <htool/blocks/admissibility_conditions.hpp>
#include <htool/blocks/blocks.hpp>
#include <htool/clustering/pca.hpp>
#include <htool/testing/geometry.hpp>
#include <random>

using namespace std;
using namespace htool;

template <class AdmissibilityCondition>
int test_blocks(int argc, char *argv[], bool symmetric) {

    int rankWorld, sizeWorld;
    MPI_Comm_size(MPI_COMM_WORLD, &sizeWorld);
    MPI_Comm_rank(MPI_COMM_WORLD, &rankWorld);

    srand(1);
    bool test = 0;

    int size = 200;
    double z = 1;
    vector<double> p(3 * size);

    srand(1);
    // we set a constant seed for rand because we want always the same result if we run the check many times
    // (two different initializations with the same seed will generate the same succession of results in the subsequent calls to rand)
    create_disk(3, z, size, p.data());

    Cluster<PCAGeometricClustering> t;
    t.build(size, p.data());

    std::unique_ptr<AdmissibilityCondition> admissibility_condition(new AdmissibilityCondition());
    Block<double> B(admissibility_condition.get(), t, t);
    B.build(symmetric);

    // Test diagonal blocks
    const Block<double> &diagonal_block = B.get_local_diagonal_block();

    if (diagonal_block.get_source_cluster().get_offset() != t.get_local_offset() && diagonal_block.get_source_cluster().get_size() != t.get_local_size() && diagonal_block.get_target_cluster().get_offset() != t.get_local_offset() && diagonal_block.get_target_cluster().get_size() != t.get_local_size()) {
        test = true;                                               // LCOV_EXCL_LINE
        std::cout << "Wrong diagonal block " << test << std::endl; // LCOV_EXCL_LINE
    }

    // Check that the whole matrix is here
    std::vector<int> represented(size * size, 0);
    const std::vector<Block<double> *> &tasks = B.get_tasks();

    for (auto block : tasks) {
        int offset_i = block->get_target_cluster().get_offset();
        int size_i   = block->get_target_cluster().get_size();
        int offset_j = block->get_source_cluster().get_offset();
        int size_j   = block->get_source_cluster().get_size();

        for (int i = offset_i; i < offset_i + size_i; i++) {
            for (int j = offset_j; j < offset_j + size_j; j++) {
                represented[i + j * size] += 1;
            }
        }
    }
    test = test || !(std::all_of(represented.begin(), represented.end(), [](int i) { return (i == 1 ? true : false); }));
    std::cout << "Full representation " << test << std::endl;

    // Check ordering of local blocks
    const std::vector<Block<double> *> &local_tasks = B.get_local_tasks();
    for (int i = 0; i < local_tasks.size() - 1; i++) {
        if (local_tasks[i]->get_target_cluster().get_offset() == local_tasks[i + 1]->get_target_cluster().get_offset()) {
            test = test || !(local_tasks[i]->get_source_cluster().get_offset() < local_tasks[i + 1]->get_source_cluster().get_offset());
        } else {
            test = test || !(local_tasks[i]->get_target_cluster().get_offset() < local_tasks[i + 1]->get_target_cluster().get_offset());
        }
    }

    if (rankWorld == 0) {
        std::cout << "test " << test << std::endl;
    }

    return test;
}
