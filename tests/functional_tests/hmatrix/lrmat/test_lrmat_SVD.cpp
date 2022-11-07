#include <complex>
#include <iostream>
#include <vector>

#include "test_lrmat.hpp"
#include <htool/clustering/pca.hpp>
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

        Cluster<PCAGeometricClustering> t, s;
        t.build(nr, xt.data());
        s.build(nc, xs.data());

        std::shared_ptr<VirtualAdmissibilityCondition> AdmissibilityCondition = std::make_shared<RjasanowSteinbach>();
        Block<double> block(AdmissibilityCondition.get(), t, s);

        GeneratorTestDouble A(3, nr, nc, xt, xs);

        // SVD fixed rank
        int reqrank_max = 10;
        SVD<double> compressor_SVD;
        LowRankMatrix<double> A_SVD_fixed(block, A, compressor_SVD, xt.data(), xs.data(), reqrank_max, epsilon);
        std::vector<double> SVD_fixed_errors;
        std::vector<double> SVD_errors_check(reqrank_max, 0);

        // compute singular values
        std::vector<double> mat(nr * nc);
        A.copy_submatrix(nr, nc, t.get_global_perm().data(), s.get_global_perm().data(), mat.data());
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

        Lapack<double>::gesvd("A", "A", &nr, &nc, mat.data(), &lda, singular_values.data(), u.data(), &ldu, vt.data(), &ldvt, work.data(), &lwork, rwork.data(), &info);
        lwork = (int)std::real(work[0]);
        work.resize(lwork);
        Lapack<double>::gesvd("A", "A", &nr, &nc, mat.data(), &lda, singular_values.data(), u.data(), &ldu, vt.data(), &ldvt, work.data(), &lwork, rwork.data(), &info);

        for (int k = 0; k < reqrank_max; k++) {
            SVD_fixed_errors.push_back(Frobenius_absolute_error(block, A_SVD_fixed, A, k));
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
        LowRankMatrix<double> A_SVD(block, A, compressor_SVD, xt.data(), xs.data(), -1, epsilon);

        std::pair<double, double> fixed_compression_interval(0.87, 0.89);
        std::pair<double, double> auto_compression_interval(0.95, 0.97);
        test = test || test_lrmat(block, A, A_SVD_fixed, A_SVD, t.get_global_perm(), s.get_global_perm(), fixed_compression_interval, auto_compression_interval, 1);
    }
    cout << "test : " << test << endl;

    // Finalize the MPI environment.
    MPI_Finalize();
    return test;
}
