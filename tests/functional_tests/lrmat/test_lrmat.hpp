#include <complex>
#include <iostream>
#include <random>
#include <vector>

#include <htool/lrmat/sympartialACA.hpp>
#include <htool/testing/generator_test.hpp>
#include <htool/testing/geometry.hpp>

using namespace std;
using namespace htool;

template <class LowRankMatrix>
int test_lrmat(const Block<double> &block, const GeneratorTestDouble &A, const LowRankMatrix &Fixed_approximation, const LowRankMatrix &Auto_approximation, const std::vector<int> &permt, const std::vector<int> &perms, std::pair<double, double> fixed_compression_interval, std::pair<double, double> auto_compression_interval, bool verbose = 0, double margin = 0) {

    bool test = 0;
    int nr    = permt.size();
    int nc    = perms.size();

    // ACA with fixed rank
    int reqrank_max = 10;
    std::vector<double> fixed_errors;
    for (int k = 0; k < Fixed_approximation.rank_of() + 1; k++) {
        fixed_errors.push_back(Frobenius_absolute_error(block, Fixed_approximation, A, k));
    }

    // Test rank
    test = test || !(Fixed_approximation.rank_of() == reqrank_max);

    if (verbose) {
        cout << "Compression with fixed rank" << endl;
        cout << "> rank : " << Fixed_approximation.rank_of() << endl;
    }

    // Test Frobenius errors
    test = test || !(fixed_errors.back() < 1e-8);

    if (verbose)
        cout << "> Errors with Frobenius norm : " << fixed_errors << endl;

    // Test compression
    test = test || !(fixed_compression_interval.first < Fixed_approximation.space_saving() && Fixed_approximation.space_saving() < fixed_compression_interval.second);
    if (verbose)
        cout << "> Compression rate : " << Fixed_approximation.space_saving() << endl;

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

    // Test mat vec prod
    std::vector<double> out_perm(nr);
    std::vector<double> f_perm(nc);
    std::vector<double> out = Fixed_approximation * f;
    for (int i = 0; i < permt.size(); i++) {
        out_perm[permt[i]] = out[i];
    }
    for (int i = 0; i < perms.size(); i++) {
        f_perm[perms[i]] = f[i];
    }
    double error = norm2(A * f_perm - out_perm) / norm2(f);
    test         = test || !(error < Fixed_approximation.get_epsilon() * (1 + margin));
    if (verbose)
        cout << "> Errors on a mat vec prod : " << error << endl;

    // ACA automatic building
    std::vector<double> auto_errors;
    for (int k = 0; k < Auto_approximation.rank_of() + 1; k++) {
        auto_errors.push_back(Frobenius_absolute_error(block, Auto_approximation, A, k));
    }

    if (verbose)
        cout << "Automatic compression" << endl;
    // Test Frobenius error
    test = test || !(auto_errors[Auto_approximation.rank_of()] < Auto_approximation.get_epsilon());
    if (verbose)
        cout << "> Errors with Frobenius norm: " << auto_errors << endl;

    // Test compression rate
    test = test || !(auto_compression_interval.first < Auto_approximation.space_saving() && Auto_approximation.space_saving() < auto_compression_interval.second);
    if (verbose)
        cout << "> Compression rate : " << Auto_approximation.space_saving() << endl;

    // Test mat vec prod
    out = Auto_approximation * f;
    for (int i = 0; i < permt.size(); i++) {
        out_perm[permt[i]] = out[i];
    }
    error = norm2(A * f_perm - out_perm) / norm2(f);
    test  = test || !(error < Auto_approximation.get_epsilon() * (1 + margin));
    if (verbose)
        cout << "> Errors on a mat vec prod : " << error << endl;
    if (verbose)
        cout << "test : " << test << endl
             << endl;
    return test;
}
