#include <htool/blocks/admissibility_conditions.hpp>
#include <htool/blocks/blocks.hpp>
#include <random>

using namespace std;
using namespace htool;

template <typename Cluster_type, template <typename> class AdmissibilityCondition>
int test_blocks(int argc, char *argv[], bool symmetric) {

    int rankWorld, sizeWorld;
    MPI_Comm_size(MPI_COMM_WORLD, &sizeWorld);
    MPI_Comm_rank(MPI_COMM_WORLD, &rankWorld);

    SetMinClusterSize(1);
    srand(1);
    bool test = 0;

    int size = 20;
    double z = 1;
    vector<R3> p(size);
    vector<double> r(size, 0);
    vector<double> g(size, 1);
    vector<int> tab(size);

    for (int j = 0; j < size; j++) {
        double rho   = ((double)rand() / (double)(RAND_MAX)); // (double) otherwise integer division!
        double theta = ((double)rand() / (double)(RAND_MAX));
        p[j][0]      = sqrt(rho) * cos(2 * M_PI * theta);
        p[j][1]      = sqrt(rho) * sin(2 * M_PI * theta);
        p[j][2]      = z;
        // sqrt(rho) otherwise the points would be concentrated in the center of the disk
        tab[j] = j;
    }

    Cluster_type t;
    t.build(p, r, tab, g);

    Block<Cluster_type, AdmissibilityCondition> B(t, t);
    B.build(symmetric);

    // Test diagonal blocks
    const Block<Cluster_type, AdmissibilityCondition> &diagonal_block = B.get_local_diagonal_block();

    if (diagonal_block.get_source_cluster().get_offset() != t.get_local_offset() && diagonal_block.get_source_cluster().get_size() != t.get_local_size() && diagonal_block.get_target_cluster().get_offset() != t.get_local_offset() && diagonal_block.get_target_cluster().get_size() != t.get_local_size()) {
        test = true;
        std::cout << "Wrong diagonal block " << test << std::endl;
    }

    // Check that the whole matrix is here
    std::vector<int> represented(size * size, 0);
    const std::vector<Block<Cluster_type, AdmissibilityCondition> *> &tasks = B.get_tasks();

    for (auto block : tasks) {
        std::cout << block->get_target_cluster().get_offset() << " " << block->get_target_cluster().get_size() << " " << block->get_source_cluster().get_offset() << " " << block->get_source_cluster().get_size() << " " << std::endl;
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
    std::cout << represented << std::endl;
    test = test || !(std::all_of(represented.begin(), represented.end(), [](int i) { return (i == 1 ? true : false); }));
    std::cout << "Full representation " << test << std::endl;

    // Check ordering of local blocks
    const std::vector<Block<Cluster_type, AdmissibilityCondition> *> &local_tasks = B.get_local_tasks();
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
