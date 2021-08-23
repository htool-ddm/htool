#include <complex>
#include <iostream>
#include <vector>

#include "test_multi_lrmat.hpp"
#include <htool/lrmat/partialACA.hpp>
#include <htool/multilrmat/multipartialACA.hpp>

using namespace std;
using namespace htool;

int main(int argc, char *argv[]) {

    // Initialize the MPI environment
    MPI_Init(&argc, &argv);

    const int ndistance = 4;
    double distance[ndistance];
    distance[0] = 15;
    distance[1] = 20;
    distance[2] = 30;
    distance[3] = 40;

    double epsilon = 0.0001;

    int nr = 500;
    int nc = 100;

    std::vector<double> xt(3 * nr);
    std::vector<double> xs(3 * nc);
    bool test = 0;

    for (int idist = 0; idist < ndistance; idist++) {

        srand(1);
        // we set a constant seed for rand because we want always the same result if we run the check many times
        // (two different initializations with the same seed will generate the same succession of results in the subsequent calls to rand)

        create_disk(3, 0, nr, xt.data());
        create_disk(3, distance[idist], nc, xs.data());

        Cluster<PCAGeometricClustering> t, s;

        t.build(nr, xt.data());
        s.build(nc, xs.data());

        MyMultiMatrix A(3, nr, nc, xt, xs);
        int nm = A.nb_matrix();
        GeneratorTestDouble A_test(3, nr, nc, xt, xs);

        // partialACA fixed rank
        int reqrank_max = 10;
        MultiLowRankMatrix<double> A_partialACA_fixed(A_test.get_dimension(), t.get_perm(), s.get_perm(), nm, reqrank_max, epsilon);
        LowRankMatrix<double> A_partialACA_fixed_test(A_test.get_dimension(), t.get_perm(), s.get_perm(), reqrank_max, epsilon);
        A_partialACA_fixed.build(A, MultipartialACA<double>(), t, xt.data(), s, xs.data());
        ;
        A_partialACA_fixed_test.build(A_test, partialACA<double>(), t, xt.data(), s, xs.data());
        ;

        // // ACA automatic building
        // MultiLowRankMatrix<double> A_partialACA(A_test.get_dimension(), t.get_perm(), s.get_perm(), nm);
        // A_partialACA.set_epsilon(epsilon);
        // A_partialACA.build(A, MultipartialACA<double>(), t, xt.data(), s, xs.data());
        // LowRankMatrix<double> A_partialACA_test(A_test.get_dimension(), t.get_perm(), s.get_perm());
        // A_partialACA_test.set_epsilon(epsilon);
        // A_partialACA_test.build(A_test, partialACA<double>(), t, xt.data(), s, xs.data());
        // ;

        // // Comparison with lrmat
        // std::vector<double> one(nc, 1);
        // test = test || !(norm2(A_partialACA_fixed[0] * one - A_partialACA_fixed_test * one) < 1e-10);
        // cout << "> Errors for fixed rank compared to lrmat: " << norm2(A_partialACA_fixed[0] * one - A_partialACA_fixed_test * one) << endl;

        // test = test || !(norm2(A_partialACA[0] * one - A_partialACA_test * one) < 1e-10);
        // cout << "> Errors for auto rank compared to lrmat: " << norm2(A_partialACA[0] * one - A_partialACA_test * one) << endl;

        // // Test multi lrmat
        // std::pair<double, double> fixed_compression_interval(0.87, 0.89);
        // std::pair<double, double> auto_compression_interval(0.93, 0.96);

        // test = test || (test_multi_lrmat(A, A_partialACA_fixed, A_partialACA, t.get_perm(), s.get_perm(), fixed_compression_interval, auto_compression_interval));
    }

    cout << "test : " << test << endl;

    // Finalize the MPI environment.
    MPI_Finalize();

    return test;
}
