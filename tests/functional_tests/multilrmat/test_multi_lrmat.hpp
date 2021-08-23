#include <complex>
#include <iostream>
#include <random>
#include <vector>

#include <htool/clustering/pca.hpp>
#include <htool/multilrmat/multilrmat.hpp>
#include <htool/testing/generator_test.hpp>
#include <htool/testing/geometry.hpp>

using namespace std;
using namespace htool;
class MyMultiMatrix : public VirtualMultiGenerator<double> {
    const vector<double> &p1;
    const vector<double> &p2;
    int space_dim;

  public:
    MyMultiMatrix(int space_dim0, int nr, int nc, int nm, const vector<double> &p10, const vector<double> &p20) : VirtualMultiGenerator(nr, nc, nm), p1(p10), p2(p20), space_dim(space_dim0) {}

    std::vector<double> get_coefs(const int &i, const int &j) const {
        return std::vector<double>{
            (1.) / (4 * M_PI * std::sqrt(std::inner_product(p1.begin() + space_dim * i, p1.begin() + space_dim * i + space_dim, p2.begin() + space_dim * j, double(0), std::plus<double>(), [](double u, double v) { return (u - v) * (u - v); }))),
            (2.) / (4 * M_PI * std::sqrt(std::inner_product(p1.begin() + space_dim * i, p1.begin() + space_dim * i + space_dim, p2.begin() + space_dim * j, double(0), std::plus<double>(), [](double u, double v) { return (u - v) * (u - v); }))),
            (3.) / (4 * M_PI * std::sqrt(std::inner_product(p1.begin() + space_dim * i, p1.begin() + space_dim * i + space_dim, p2.begin() + space_dim * j, double(0), std::plus<double>(), [](double u, double v) { return (u - v) * (u - v); }))),
            (4.) / (4 * M_PI * std::sqrt(std::inner_product(p1.begin() + space_dim * i, p1.begin() + space_dim * i + space_dim, p2.begin() + space_dim * j, double(0), std::plus<double>(), [](double u, double v) { return (u - v) * (u - v); }))),
            (5.) / (4 * M_PI * std::sqrt(std::inner_product(p1.begin() + space_dim * i, p1.begin() + space_dim * i + space_dim, p2.begin() + space_dim * j, double(0), std::plus<double>(), [](double u, double v) { return (u - v) * (u - v); })))};
    }

    void copy_submatrices(int M, int N, const int *const rows, const int *const cols, int NM, double *ptr) const override {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                ptr[i + M * j] = this->get_coef(rows[i], cols[j]);
            }
        }
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

template <class MultiLowRankMatrix>
int test_multi_lrmat(const MyMultiMatrix &A, const MultiLowRankMatrix &Fixed_approximation, const MultiLowRankMatrix &Auto_approximation, const std::vector<int> &permt, const std::vector<int> &perms, std::pair<double, double> fixed_compression_interval, std::pair<double, double> auto_compression_interval) {

    bool test = 0;
    int nr    = permt.size();
    int nc    = perms.size();

    // Random vector
    double lower_bound = 0;
    double upper_bound = 10000;
    std::random_device rd;
    std::mt19937 mersenne_engine(rd());
    std::uniform_real_distribution<double> dist(lower_bound, upper_bound);
    auto gen = [&dist, &mersenne_engine]() {
        return dist(mersenne_engine);
    };

    vector<double> f(nc, 1);
    generate(begin(f), end(f), gen);

    // ACA with fixed rank
    int reqrank_max = 10;
    std::vector<std::vector<double>> fixed_errors;
    for (int k = 0; k < reqrank_max + 1; k++) {
        fixed_errors.push_back(Frobenius_absolute_error(Fixed_approximation, A, k));
    }

    // Test rank
    test = test || !(Fixed_approximation.rank_of() == reqrank_max);
    std::cout << test << " " << reqrank_max << " " << Fixed_approximation.rank_of() << std::endl;
    cout << "Compression with fixed rank" << endl;
    cout << "> rank : " << Fixed_approximation.rank_of() << endl;

    // Test Frobenius errors
    test = test || !(max(fixed_errors.back()) < 1e-7);
    cout << "> Errors with Frobenius norm : " << fixed_errors << endl;

    for (int l = 0; l < A.nb_matrix(); l++) {

        // Test compression
        test = test || !(fixed_compression_interval.first < Fixed_approximation[l].space_saving() && Fixed_approximation[l].space_saving() < fixed_compression_interval.second);
        cout << "> Compression rate : " << Fixed_approximation[l].space_saving() << endl;

        // Test mat vec prod
        std::vector<double> out_perm(nr);
        std::vector<double> out = Fixed_approximation[l] * f;
        for (int i = 0; i < permt.size(); i++) {
            out_perm[permt[i]] = out[i];
        }
        double error = norm2(A.mult(f, l) - out_perm) / norm2(A.mult(f, l));
        test         = test || !(error < Fixed_approximation[l].get_epsilon() * 10);
        cout << "> Errors on a mat vec prod : " << error << " " << (Fixed_approximation[l].get_epsilon() * 10) << " " << (error < Fixed_approximation[l].get_epsilon() * 10) << endl;
        cout << "test : " << test << endl
             << endl;
    }

    // ACA automatic building
    std::vector<std::vector<double>> auto_errors;
    for (int k = 0; k < Auto_approximation.rank_of() + 1; k++) {
        auto_errors.push_back(Frobenius_absolute_error(Auto_approximation, A, k));
    }
    cout << "Automatic compression" << endl;

    // Test Frobenius error
    test = test || !(max(auto_errors[Auto_approximation.rank_of()]) < Auto_approximation.get_epsilon());
    cout << "> Errors with Frobenius norm: " << auto_errors << endl;

    for (int l = 0; l < A.nb_matrix(); l++) {

        // Test compression rate
        test = test || !(auto_compression_interval.first < Auto_approximation[l].space_saving() && Auto_approximation[l].space_saving() < auto_compression_interval.second);
        cout << "> Compression rate : " << Auto_approximation[l].space_saving() << endl;

        // Test mat vec prod
        std::vector<double> out_perm(nr);
        std::vector<double> out = Auto_approximation[l] * f;
        for (int i = 0; i < permt.size(); i++) {
            out_perm[permt[i]] = out[i];
        }
        double error = norm2(A.mult(f, l) - out_perm) / norm2(A.mult(f, l));
        test         = test || !(error < Auto_approximation[l].get_epsilon() * 10);
        cout << "> Errors on a mat vec prod : " << error << endl;

        cout << "test : " << test << endl
             << endl;
    }

    return test;
}
