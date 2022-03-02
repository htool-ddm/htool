#include <complex>
#include <iostream>
#include <random>
#include <vector>

#include <htool/clustering/pca.hpp>
#include <htool/lrmat/SVD.hpp>
#include <htool/lrmat/partialACA.hpp>
#include <htool/testing/generator_test.hpp>
#include <htool/testing/geometry.hpp>
#include <htool/types/hmatrix.hpp>

using namespace std;
using namespace htool;

int main(int argc, char *argv[]) {

    // Initialize the MPI environment
    MPI_Init(&argc, &argv);

    // Get the number of processes
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Get the rank of the process
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    //
    bool test           = 0;
    const int ndistance = 4;
    double distance[ndistance];
    distance[0] = 3;
    distance[1] = 5;
    distance[2] = 7;
    distance[3] = 10;

    double epsilon = 1e-3;
    double eta     = 0.1;

    for (int idist = 0; idist < ndistance; idist++) {
        // cout << "Distance between the clusters: " << distance[idist] << endl;

        srand(1);
        // we set a constant seed for rand because we want always the same result if we run the check many times
        // (two different initializations with the same seed will generate the same succession of results in the subsequent calls to rand)

        int nr = 2000;
        int nc = 1000;

        double z1 = 1;
        double z2 = 1 + distance[idist];
        vector<double> p1(3 * nr);
        vector<double> p2(3 * nc);

        srand(1);
        // we set a constant seed for rand because we want always the same result if we run the check many times
        // (two different initializations with the same seed will generate the same succession of results in the subsequent calls to rand)
        create_disk(3, z1, nr, p1.data());
        create_disk(3, z2, nc, p2.data());

        int size_numbering = nr / (size);
        int count_size     = 0;
        std::vector<int> MasterOffset_target;
        for (int p = 0; p < size - 1; p++) {
            MasterOffset_target.push_back(count_size);
            MasterOffset_target.push_back(size_numbering);

            count_size += size_numbering;
        }
        MasterOffset_target.push_back(count_size);
        MasterOffset_target.push_back(nr - count_size);

        size_numbering = nc / size;
        count_size     = 0;

        std::vector<int> MasterOffset_source;
        for (int p = 0; p < size - 1; p++) {
            MasterOffset_source.push_back(count_size);
            MasterOffset_source.push_back(size_numbering);

            count_size += size_numbering;
        }
        MasterOffset_source.push_back(count_size);
        MasterOffset_source.push_back(nc - count_size);

        vector<double> rhs(p2.size(), 1);
        GeneratorTestDouble A(3, nr, nc, p1, p2);
        std::shared_ptr<Cluster<PCARegularClustering>> t = make_shared<Cluster<PCARegularClustering>>();
        std::shared_ptr<Cluster<PCARegularClustering>> s = make_shared<Cluster<PCARegularClustering>>();
        t->build(nr, p1.data(), MasterOffset_target.data(), 2);
        s->build(nc, p2.data(), MasterOffset_source.data(), 2);

        // with permutation
        std::shared_ptr<partialACA<double>> compressor = std::make_shared<partialACA<double>>();
        HMatrix<double> HA(t, s, epsilon, eta);
        HA.set_compression(compressor);
        HA.build(A, p1.data(), p2.data());
        HA.print_infos();

        // without permutation
        vector<double> p1_perm(3 * nr);
        vector<double> p2_perm(3 * nc);
        for (int i = 0; i < t->get_global_perm().size(); i++) {
            p1_perm[i * 3 + 0] = p1[t->get_global_perm(i) * 3 + 0];
            p1_perm[i * 3 + 1] = p1[t->get_global_perm(i) * 3 + 1];
            p1_perm[i * 3 + 2] = p1[t->get_global_perm(i) * 3 + 2];
        }
        for (int i = 0; i < s->get_global_perm().size(); i++) {
            p2_perm[i * 3 + 0] = p2[s->get_global_perm(i) * 3 + 0];
            p2_perm[i * 3 + 1] = p2[s->get_global_perm(i) * 3 + 1];
            p2_perm[i * 3 + 2] = p2[s->get_global_perm(i) * 3 + 2];
        }

        GeneratorTestDouble A_perm(3, nr, nc, p1_perm, p2_perm);
        HMatrix<double> HA_not_using_perm(t, s, epsilon, eta);
        HA_not_using_perm.set_compression(compressor);
        HA_not_using_perm.set_use_permutation(false);
        HA_not_using_perm.build(A_perm, p1_perm.data(), p2_perm.data());

        // Random vector
        int mu = 10;
        vector<double> f(nc * mu, 1);
        if (rank == 0) {
            double lower_bound = 0;
            double upper_bound = 10000;
            std::random_device rd;
            std::mt19937 mersenne_engine(rd());
            std::uniform_real_distribution<double> dist(lower_bound, upper_bound);
            auto gen = [&dist, &mersenne_engine]() {
                return dist(mersenne_engine);
            };

            generate(begin(f), end(f), gen);
        }
        MPI_Bcast(f.data(), nc * mu, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // Global with permutation
        std::vector<double> f_vec(f.begin(), f.begin() + nc);
        std::vector<double>
            result_vec_global_to_global(nr, 0), result_global_to_global(nr * mu, 0);

        HA.mvprod_global_to_global(f_vec.data(), result_vec_global_to_global.data());
        HA.mvprod_global_to_global(f.data(), result_global_to_global.data(), mu);

        // Local with permutation
        std::vector<double> f_vec_local(f.begin() + MasterOffset_source[2 * rank], f.begin() + MasterOffset_source[2 * rank] + MasterOffset_source[2 * rank + 1]);
        std::vector<double> f_local(MasterOffset_source[2 * rank + 1] * mu);
        for (int i = 0; i < mu; i++) {
            std::copy_n(f.data() + MasterOffset_source[2 * rank] + nc * i, MasterOffset_source[2 * rank + 1], f_local.data() + MasterOffset_source[2 * rank + 1] * i);
        }
        std::vector<double> result_vec_local_to_local(MasterOffset_target[2 * rank + 1], 0), result_local_to_local(MasterOffset_target[2 * rank + 1] * mu);

        HA.mvprod_local_to_local(f_vec_local.data(), result_vec_local_to_local.data());
        HA.mvprod_local_to_local(f_local.data(), result_local_to_local.data(), mu);

        // Error between local and global with permutation
        double global_local_diff     = 0;
        double global_vec_local_diff = 0;
        double norm2_global          = norm2(result_vec_global_to_global);
        for (int i = 0; i < MasterOffset_target[2 * rank + 1]; i++) {
            global_vec_local_diff += std::pow(result_vec_global_to_global[i + MasterOffset_target[2 * rank]] - result_vec_local_to_local[i], 2);
            for (int j = 0; j < mu; j++) {
                global_local_diff += std::pow(result_global_to_global[i + MasterOffset_target[2 * rank] + j * nr] - result_local_to_local[i + j * MasterOffset_target[2 * rank + 1]], 2);
            }
        }

        test = test || !(std::sqrt(global_vec_local_diff) / norm2_global < 1e-10);
        test = test || !(std::sqrt(global_local_diff) / norm2_global < 1e-10);

        if (rank == 0) {
            cout << "difference on mat vec prod computed globally and locally: " << std::sqrt(global_vec_local_diff) / norm2_global << endl;
            cout << "difference on mat mat prod computed globally and locally: " << std::sqrt(global_local_diff) / norm2_global << endl;
            cout << "test: " << test << endl;
        }

        // Global without permutation
        std::vector<double> f_vec_perm(nc, 0), f_perm(nc * mu, 0);
        HA.source_to_cluster_permutation(f.data(), f_vec_perm.data());
        for (int i = 0; i < mu; i++) {
            HA.source_to_cluster_permutation(f.data() + nc * i, f_perm.data() + nc * i);
        }
        std::vector<double>
            result_vec_global_to_global_perm(nr, 0), result_global_to_global_perm(nr * mu, 0), result_vec_global_to_global_2(nr, 0), result_global_to_global_2(nr * mu, 0);

        HA_not_using_perm.mvprod_global_to_global(f_vec_perm.data(), result_vec_global_to_global_perm.data());
        HA_not_using_perm.mvprod_global_to_global(f_perm.data(), result_global_to_global_perm.data(), mu);

        HA.cluster_to_target_permutation(result_vec_global_to_global_perm.data(), result_vec_global_to_global_2.data());
        for (int i = 0; i < mu; i++) {
            HA.cluster_to_target_permutation(result_global_to_global_perm.data() + nr * i, result_global_to_global_2.data() + nr * i);
        }

        // Erreur global
        double erreur_global_vec2 = norm2(result_vec_global_to_global_2 - result_vec_global_to_global) / norm2(result_vec_global_to_global);
        double erreur_global2     = norm2(result_global_to_global_2 - result_global_to_global) / norm2(result_global_to_global);

        test = test || !(erreur_global_vec2 < 1e-10);
        test = test || !(erreur_global2 < 1e-10);

        if (rank == 0) {
            cout << "Global errors on a mat vec prod : " << erreur_global_vec2 << endl;
            cout << "Global errors on a mat mat prod : " << erreur_global2 << endl;
            cout << "test: " << test << endl;
        }

        // Local without permutation
        std::vector<double> f_vec_local_perm(f_vec_local.size()), f_local_perm(f_local.size());
        HA.local_source_to_local_cluster(f_vec_local.data(), f_vec_local_perm.data());
        for (int i = 0; i < mu; i++) {
            HA.local_source_to_local_cluster(f_local.data() + MasterOffset_source[2 * rank + 1] * i, f_local_perm.data() + MasterOffset_source[2 * rank + 1] * i);
        }

        std::vector<double> result_vec_local_to_local_perm(MasterOffset_target[2 * rank + 1], 0), result_local_to_local_perm(MasterOffset_target[2 * rank + 1] * mu), result_vec_local_to_local_2(MasterOffset_target[2 * rank + 1], 0), result_local_to_local_2(MasterOffset_target[2 * rank + 1] * mu);

        HA_not_using_perm.mvprod_local_to_local(f_vec_local_perm.data(), result_vec_local_to_local_perm.data());
        HA_not_using_perm.mvprod_local_to_local(f_local_perm.data(), result_local_to_local_perm.data(), mu);

        HA.local_cluster_to_local_target(result_vec_local_to_local_perm.data(), result_vec_local_to_local_2.data());
        for (int i = 0; i < mu; i++) {
            HA.local_cluster_to_local_target(result_local_to_local_perm.data() + MasterOffset_target[2 * rank + 1] * i, result_local_to_local_2.data() + MasterOffset_target[2 * rank + 1] * i);
        }

        // Erreur local
        double erreur_local_vec2 = norm2(result_vec_local_to_local_2 - result_vec_local_to_local) / norm2(result_vec_local_to_local);
        double erreur_local2     = norm2(result_local_to_local_2 - result_local_to_local) / norm2(result_local_to_local);

        test = test || !(erreur_local_vec2 < 1e-10);
        test = test || !(erreur_local2 < 1e-10);

        if (rank == 0) {
            cout << "Local errors on a mat vec prod : " << erreur_local_vec2 << endl;
            cout << "Local errors on a mat mat prod : " << erreur_local2 << endl;
            cout << "test: " << test << endl;
        }
    }

    // Finalize the MPI environment.
    MPI_Finalize();
    return test;
}
