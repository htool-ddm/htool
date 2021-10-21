
#ifndef HTOOL_TESTING_GENERATOR_INPUT_HPP
#define HTOOL_TESTING_GENERATOR_INPUT_HPP
#include <complex>
#include <random>
#include <vector>
namespace htool {

template <typename T>
void generate_random_vector(std::vector<T> &random_vector) {
    T lower_bound = 0;
    T upper_bound = 10000;
    std::random_device rd;
    std::mt19937 mersenne_engine(rd());
    std::uniform_real_distribution<T> dist(lower_bound, upper_bound);
    auto gen = [&dist, &mersenne_engine]() {
        return dist(mersenne_engine);
    };

    generate(begin(random_vector), end(random_vector), gen);
}

template <typename T>
void generate_random_vector(std::vector<std::complex<T>> &random_vector) {
    std::vector<T> random_vector_real(random_vector.size()), random_vector_imag(random_vector.size());
    T lower_bound = 0;
    T upper_bound = 10000;
    std::random_device rd;
    std::mt19937 mersenne_engine(rd());
    std::uniform_real_distribution<T> dist(lower_bound, upper_bound);
    auto gen = [&dist, &mersenne_engine]() {
        return dist(mersenne_engine);
    };

    generate(begin(random_vector_real), end(random_vector_real), gen);
    generate(begin(random_vector_imag), end(random_vector_imag), gen);
    std::transform(random_vector_real.begin(), random_vector_real.end(), random_vector_imag.begin(), random_vector.begin(), [](double da, double db) {
        return std::complex<double>(da, db);
    });
}

} // namespace htool

#endif
