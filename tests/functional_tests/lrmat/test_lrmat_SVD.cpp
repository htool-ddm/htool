#include <complex>
#include <iostream>
#include <vector>

#include "test_lrmat.hpp"
#include <htool/clustering/ncluster.hpp>
#include <htool/lrmat/SVD.hpp>

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

        GeometricClustering t, s;
        t.build_global_auto(nr, xt.data());
        s.build_global_auto(nc, xs.data());

        IMatrixTestDouble A(3, nr, nc, xt, xs);

        // SVD fixed rank
        int reqrank_max = 10;
        SVD<double, GeometricClustering> A_SVD_fixed(t.get_perm(), s.get_perm(), reqrank_max, epsilon);
        A_SVD_fixed.build(A);
        std::vector<double> SVD_fixed_errors;
        std::vector<double> SVD_errors_check(reqrank_max, 0);

        for (int k = 0; k < reqrank_max; k++) {
            SVD_fixed_errors.push_back(Frobenius_absolute_error(A_SVD_fixed, A, k));
            for (int l = k; l < min(nr, nc); l++) {
                SVD_errors_check[k] += pow(A_SVD_fixed.get_singular_value(l), 2);
            }
            SVD_errors_check[k] = sqrt(SVD_errors_check[k]);
        }

        // Testing with Eckart–Young–Mirsky theorem for Frobenius norm
        cout << "Testing with Eckart–Young–Mirsky theorem" << endl;
        test = test || !(norm2(SVD_fixed_errors - SVD_errors_check) < 1e-10);
        cout << "> Errors with Frobenius norm: " << SVD_fixed_errors << endl;
        cout << "> Errors computed with the remaining eigenvalues : " << SVD_errors_check << endl;

        // ACA automatic building
        SVD<double, GeometricClustering> A_SVD(t.get_perm(), s.get_perm());
        A_SVD.set_epsilon(epsilon);
        A_SVD.build(A);

        std::pair<double, double> fixed_compression_interval(0.87, 0.89);
        std::pair<double, double> auto_compression_interval(0.95, 0.97);
        test = test || test_lrmat(A, A_SVD_fixed, A_SVD, t.get_perm(), s.get_perm(), fixed_compression_interval, auto_compression_interval, 1);
    }
    cout << "test : " << test << endl;

    // Finalize the MPI environment.
    MPI_Finalize();
    return test;
}
