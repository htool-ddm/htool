#include <complex>
#include <iostream>
#include <vector>

#include "../test_lrmat_build.hpp"
#include <htool/clustering/clustering.hpp>
#include <htool/hmatrix/lrmat/SVD.hpp>
#include <htool/testing/geometry.hpp>
#include <mpi.h>

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

        create_disk(3, 0., nr, xt.data());
        create_disk(3, distance[idist], nc, xs.data());

        ClusterTreeBuilder<double> recursive_build_strategy;

        Cluster<double> t = recursive_build_strategy.create_cluster_tree(nr, 3, xt.data(), 2, 2);
        Cluster<double> s = recursive_build_strategy.create_cluster_tree(nc, 3, xs.data(), 2, 2);

        GeneratorTestDouble A(3, nr, nc, xt, xs, t, s, true, true);

        // SVD fixed rank
        int reqrank_max = 10;
        SVD<double> compressor_SVD;
        LowRankMatrix<double> A_SVD_fixed(A, compressor_SVD, t, s, reqrank_max, epsilon);
        std::vector<double> SVD_fixed_errors;
        std::vector<double> SVD_errors_check(reqrank_max, 0);

        // compute singular values
        Matrix<double> matrix(nr, nc);
        A.copy_submatrix(t.get_size(), s.get_size(), t.get_offset(), s.get_offset(), matrix.data());
        int lda   = nr;
        int ldu   = nr;
        int ldvt  = nc;
        int lwork = -1;
        int info;
        std::vector<double> singular_values(std::min(nr, nc));
        Matrix<double> u(nr, nr);
        // std::vector<T> vt (n*n);
        Matrix<double> vt(nc, nc);
        std::vector<double> work(std::min(nc, nr));
        std::vector<double> rwork(5 * std::min(nr, nc));

        Lapack<double>::gesvd("A", "A", &nr, &nc, matrix.data(), &lda, singular_values.data(), u.data(), &ldu, vt.data(), &ldvt, work.data(), &lwork, rwork.data(), &info);
        lwork = (int)std::real(work[0]);
        work.resize(lwork);
        Lapack<double>::gesvd("A", "A", &nr, &nc, matrix.data(), &lda, singular_values.data(), u.data(), &ldu, vt.data(), &ldvt, work.data(), &lwork, rwork.data(), &info);

        for (int k = 0; k < reqrank_max; k++) {
            SVD_fixed_errors.push_back(Frobenius_absolute_error(t, s, A_SVD_fixed, A, k));
            for (int l = k; l < min(nr, nc); l++) {
                SVD_errors_check[k] += singular_values[l] * singular_values[l];
            }
            SVD_errors_check[k] = sqrt(SVD_errors_check[k]);
        }

        // Testing with Eckart–Young–Mirsky theorem for Frobenius norm
        cout << "Testing with Eckart–Young–Mirsky theorem" << endl;
        test = test || !(norm2(SVD_fixed_errors - SVD_errors_check) < 1e-10);
        cout << "> Errors with Frobenius norm: " << SVD_fixed_errors << endl;
        cout << "> Errors computed with the remaining eigenvalues : " << SVD_errors_check << endl;

        // ACA automatic building
        LowRankMatrix<double> A_SVD(A, compressor_SVD, t, s, -1, epsilon);

        std::pair<double, double> fixed_compression_interval(0.87, 0.89);
        std::pair<double, double> auto_compression_interval(0.95, 0.97);
        test = test || test_lrmat(t, s, A, A_SVD_fixed, A_SVD, fixed_compression_interval, auto_compression_interval);
    }
    cout << "test : " << test << endl;

    // Finalize the MPI environment.
    MPI_Finalize();
    return test;
}
