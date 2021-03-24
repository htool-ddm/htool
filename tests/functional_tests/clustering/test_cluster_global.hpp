#include <htool/clustering/cluster.hpp>
#include <htool/testing/geometry.hpp>
#include <random>

using namespace std;
using namespace htool;

template <typename Cluster_type, int dim>
int test_cluster_global(int argc, char *argv[]) {

    int rankWorld, sizeWorld;
    MPI_Comm_size(MPI_COMM_WORLD, &sizeWorld);
    MPI_Comm_rank(MPI_COMM_WORLD, &rankWorld);

    srand(1);
    bool test = 0;

    int size = 200;
    vector<double> p(size * dim);
    vector<double> r(size, 0);
    vector<double> g(size, 1);
    vector<int> tab(size);

    srand(1);
    // we set a constant seed for rand because we want always the same result if we run the check many times
    // (two different initializations with the same seed will generate the same succession of results in the subsequent calls to rand)
    create_disk(dim, 0, size, p.data(), tab.data());

    std::vector<int> nb_sons_test{2, 4, -1};
    for (auto &nb_sons : nb_sons_test) {
        if (rankWorld == 0) {
            cout << "Number of sons : " << nb_sons << endl;
        }

        Cluster_type t(dim);
        t.set_minclustersize(1);
        t.build_global(size, p.data(), r.data(), tab.data(), g.data(), nb_sons);
        t.print();
        MPI_Barrier(MPI_COMM_WORLD);

        // Testing recursivity
        std::stack<Cluster_type *> s;
        s.push(&t);
        int depth = 0;
        while (!s.empty()) {
            Cluster_type *curr = s.top();
            s.pop();
            if (!curr->IsLeaf()) {
                // test num inclusion

                int count = 0;
                for (int l = 0; l < curr->get_nb_sons(); l++) {
                    test = test || !(curr->get_offset() + count == curr->get_son(l).get_offset());
                    count += curr->get_son(l).get_size();
                }

                for (int l = 0; l < curr->get_nb_sons(); l++) {
                    s.push((&(curr->get_son(l))));
                }
            }
        }

        // Testing getters for local cluster
        int local_size   = t.get_local_size();
        int local_offset = t.get_local_offset();

        // Testing getters for root cluster
        int root_size   = t.get_root().get_size();
        int root_offset = t.get_root().get_offset();
        test            = test || !(root_size == size);
        test            = test || !(root_offset == 0);

        // Testing to get local cluster
        std::shared_ptr<Cluster_type> local_cluster = t.get_local_cluster_tree();
        test                                        = test || !(local_size == local_cluster->get_size());
        test                                        = test || !(local_offset == local_cluster->get_offset());
        std::stack<Cluster_type const *> s_local_1;
        std::stack<Cluster_type const *> s_local_2;
        s_local_1.push(local_cluster.get());
        s_local_2.push(&(t.get_local_cluster()));
        depth = 0;
        while (!s_local_1.empty()) {
            Cluster_type const *curr_1 = s_local_1.top();
            Cluster_type const *curr_2 = s_local_2.top();
            s_local_1.pop();
            s_local_2.pop();

            test = test || !(curr_1->get_offset() == curr_2->get_offset());
            test = test || !(curr_1->get_size() == curr_2->get_size());

            if (!curr_2->IsLeaf()) {
                // test num inclusion

                for (int l = 0; l < curr_2->get_nb_sons(); l++) {
                    s_local_1.push(&(curr_1->get_son(l)));
                    s_local_2.push(&(curr_2->get_son(l)));
                }
            }
        }

        // Testing save and read cluster
        t.save_cluster("test_cluster");
        MPI_Barrier(MPI_COMM_WORLD);
        Cluster_type copy_t(dim);
        copy_t.read_cluster("test_cluster_permutation.csv", "test_cluster_tree.csv");

        std::stack<Cluster_type const *> s_save;
        std::stack<Cluster_type const *> s_read;
        s_save.push(&t);
        s_read.push(&(copy_t));
        while (!s_save.empty()) {
            Cluster_type const *curr_1 = s_save.top();
            Cluster_type const *curr_2 = s_read.top();
            s_save.pop();
            s_read.pop();

            test = test || !(curr_1->get_offset() == curr_2->get_offset());
            test = test || !(curr_1->get_size() == curr_2->get_size());

            if (!curr_2->IsLeaf()) {
                // test num inclusion

                for (int l = 0; l < curr_2->get_nb_sons(); l++) {
                    s_save.push(&(curr_1->get_son(l)));
                    s_read.push(&(curr_2->get_son(l)));
                }
            }
        }

        // Random vector
        double lower_bound = 0;
        double upper_bound = 10000;
        std::random_device rd;
        std::mt19937 mersenne_engine(rd());
        std::uniform_real_distribution<double> dist(lower_bound, upper_bound);
        auto gen = [&dist, &mersenne_engine]() {
            return dist(mersenne_engine);
        };

        std::vector<double> random_vector_in(size), temp(size), random_vector_out(size);
        generate(begin(random_vector_in), end(random_vector_in), gen);
        t.cluster_to_global(random_vector_in.data(), temp.data());
        t.global_to_cluster(temp.data(), random_vector_out.data());

        // Test permutations
        test = test || !(norm2(random_vector_in - random_vector_out) < 1e-16);

        // Test access to local clusters
        if (sizeWorld > 1) {
            test = test || !(t.get_local_cluster().get_rank() == rankWorld);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        if (rankWorld == 0) {
            cout << "max depth : " << t.get_max_depth() << endl;
            cout << "min depth : " << t.get_min_depth() << endl;
        }
        test = test || !(t.get_max_depth() >= t.get_min_depth() && t.get_min_depth() >= 0);
    }

    if (rankWorld == 0) {
        std::cout << "test global " << test << std::endl;
    }

    return test;
}
