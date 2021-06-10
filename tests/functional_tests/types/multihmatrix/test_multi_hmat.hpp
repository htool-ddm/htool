#include <complex>
#include <iostream>
#include <random>
#include <vector>

#include <htool/clustering/pca.hpp>
#include <htool/lrmat/SVD.hpp>
#include <htool/lrmat/fullACA.hpp>
#include <htool/lrmat/partialACA.hpp>
#include <htool/lrmat/sympartialACA.hpp>
#include <htool/multilrmat/multipartialACA.hpp>
#include <htool/types/multihmatrix.hpp>

using namespace std;
using namespace htool;

class MyMultiMatrix : public MultiIMatrix<double> {
    const vector<double> &p1;
    const vector<double> &p2;
    int space_dim;

  public:
    MyMultiMatrix(int space_dim0, int nr, int nc, const vector<double> &p10, const vector<double> &p20) : MultiIMatrix(nr, nc, 5), p1(p10), p2(p20), space_dim(space_dim0) {}
    std::vector<double> get_coefs(const int &i, const int &j) const {
        return std::vector<double>{
            (1.) / (4 * M_PI * std::sqrt(std::inner_product(p1.begin() + space_dim * i, p1.begin() + space_dim * i + space_dim, p2.begin() + space_dim * j, double(0), std::plus<double>(), [](double u, double v) { return (u - v) * (u - v); }))),
            (2.) / (4 * M_PI * std::sqrt(std::inner_product(p1.begin() + space_dim * i, p1.begin() + space_dim * i + space_dim, p2.begin() + space_dim * j, double(0), std::plus<double>(), [](double u, double v) { return (u - v) * (u - v); }))),
            (3.) / (4 * M_PI * std::sqrt(std::inner_product(p1.begin() + space_dim * i, p1.begin() + space_dim * i + space_dim, p2.begin() + space_dim * j, double(0), std::plus<double>(), [](double u, double v) { return (u - v) * (u - v); }))),
            (4.) / (4 * M_PI * std::sqrt(std::inner_product(p1.begin() + space_dim * i, p1.begin() + space_dim * i + space_dim, p2.begin() + space_dim * j, double(0), std::plus<double>(), [](double u, double v) { return (u - v) * (u - v); }))),
            (5.) / (4 * M_PI * std::sqrt(std::inner_product(p1.begin() + space_dim * i, p1.begin() + space_dim * i + space_dim, p2.begin() + space_dim * j, double(0), std::plus<double>(), [](double u, double v) { return (u - v) * (u - v); })))};
    }
    std::vector<double> mult(std::vector<double> &a, int l) const {
        std::vector<double> result(this->nr, 0);
        for (int i = 0; i < this->nr; i++) {
            for (int k = 0; k < this->nc; k++) {
                result[i] += this->get_coefs(i, k)[l] * a[k];
            }
        }
        return result;
    }

    double normFrob(int l) const {
        double norm = 0;
        for (int j = 0; j < this->nb_rows(); j++) {
            for (int k = 0; k < this->nb_cols(); k++) {
                norm = norm + std::pow(std::abs((this->get_coefs(j, k))[l]), 2);
            }
        }
        return sqrt(norm);
    }
};

template <template <typename> class MultiLowRankMatrix>
int test_multi_hmat_cluster(const MyMultiMatrix &MultiA, const MultiHMatrix<double, MultiLowRankMatrix, RjasanowSteinbach> &MultiHA, int l) {
    bool test = 0;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::vector<double> f(MultiA.nb_cols(), 1);
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
    MPI_Bcast(f.data(), MultiA.nb_cols(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    std::vector<double> result(MultiA.nb_rows(), 0);
    MultiHA[l].print_infos();
    result            = MultiHA[l] * f;
    double erreur2    = norm2(MultiA.mult(f, l) - result) / norm2(MultiA.mult(f, l));
    double erreurFrob = Frobenius_absolute_error(MultiHA, MultiA, l) / MultiA.normFrob(l);

    test = test || !(erreurFrob < MultiHA.get_epsilon());
    test = test || !(erreur2 < MultiHA.get_epsilon());

    if (rank == 0) {
        cout << "Errors with Frobenius norm: " << erreurFrob << endl;
        cout << "Errors on a mat vec prod : " << erreur2 << endl;
        cout << "test: " << test << endl;
    }

    return test;
}
