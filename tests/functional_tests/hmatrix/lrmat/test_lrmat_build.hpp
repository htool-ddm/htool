#include <complex>
#include <iostream>
#include <random>
#include <vector>

#include <htool/clustering/cluster_node.hpp>
#include <htool/testing/generator_input.hpp>
#include <htool/testing/generator_test.hpp>
#include <htool/testing/geometry.hpp>

using namespace std;
using namespace htool;

template <class LowRankMatrix>
bool test_lrmat(const Cluster<double> &target_cluster, const Cluster<double> &source_cluster, const GeneratorTestDouble &A, const LowRankMatrix &Fixed_approximation, const LowRankMatrix &Auto_approximation, std::pair<double, double> fixed_compression_interval, std::pair<double, double> auto_compression_interval) {

    bool test = 0;

    // ACA with fixed rank
    int reqrank_max = 10;
    std::vector<double> fixed_errors;
    for (int k = 0; k < Fixed_approximation.rank_of() + 1; k++) {
        fixed_errors.push_back(Frobenius_absolute_error(target_cluster, source_cluster, Fixed_approximation, A, k));
    }

    // Test rank
    test = test || !(Fixed_approximation.rank_of() == reqrank_max);
    cout << "Compression with fixed rank" << endl;
    cout << "> rank : " << Fixed_approximation.rank_of() << endl;

    // Test Frobenius errors
    test = test || !(fixed_errors.back() < 1e-8);
    cout << "> Errors with Frobenius norm : " << fixed_errors << endl;

    // Test compression
    test = test || !(fixed_compression_interval.first < Fixed_approximation.space_saving() && Fixed_approximation.space_saving() < fixed_compression_interval.second);
    cout << "> Compression rate : " << Fixed_approximation.space_saving() << endl;

    // ACA automatic building
    std::vector<double> auto_errors;
    for (int k = 0; k < Auto_approximation.rank_of() + 1; k++) {
        auto_errors.push_back(Frobenius_absolute_error(target_cluster, source_cluster, Auto_approximation, A, k));
    }

    cout << "Automatic compression" << endl;
    // Test Frobenius error
    test = test || !(auto_errors[Auto_approximation.rank_of()] < Auto_approximation.get_epsilon());
    cout << "> Errors with Frobenius norm: " << auto_errors << endl;

    // Test compression rate
    test = test || !(auto_compression_interval.first < Auto_approximation.space_saving() && Auto_approximation.space_saving() < auto_compression_interval.second);
    cout << "> Compression rate : " << Auto_approximation.space_saving() << endl;

    return test;
}
