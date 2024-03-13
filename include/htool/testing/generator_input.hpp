
#ifndef HTOOL_TESTING_GENERATOR_INPUT_HPP
#define HTOOL_TESTING_GENERATOR_INPUT_HPP
#include <algorithm>
#include <complex>
#include <random>
#include <vector>
namespace htool {

template <typename T>
void generate_random_array(T *ptr, size_t size) {
    T lower_bound = 0;
    T upper_bound = 1000;
    std::random_device rd;
    std::mt19937 mersenne_engine(rd());
    std::uniform_real_distribution<T> dist(lower_bound, upper_bound);
    auto gen = [&dist, &mersenne_engine]() {
        return dist(mersenne_engine);
    };

    std::generate(ptr, ptr + size, gen);
}
template <>
void generate_random_array(int *ptr, size_t size) {
    int lower_bound = 0;
    int upper_bound = 10000;
    std::random_device rd;
    std::mt19937 mersenne_engine(rd());
    std::uniform_int_distribution<int> dist(lower_bound, upper_bound);
    auto gen = [&dist, &mersenne_engine]() {
        return dist(mersenne_engine);
    };

    std::generate(ptr, ptr + size, gen);
}

template <typename T>
void generate_random_array(std::complex<T> *ptr, size_t size) {
    std::vector<T> random_vector_real(size), random_vector_imag(size);
    T lower_bound = 0;
    T upper_bound = 10000;
    std::random_device rd;
    std::mt19937 mersenne_engine(rd());
    std::uniform_real_distribution<T> dist(lower_bound, upper_bound);
    auto gen = [&dist, &mersenne_engine]() {
        return dist(mersenne_engine);
    };

    std::generate(begin(random_vector_real), end(random_vector_real), gen);
    std::generate(begin(random_vector_imag), end(random_vector_imag), gen);
    std::transform(random_vector_real.begin(), random_vector_real.end(), random_vector_imag.begin(), ptr, [](double da, double db) {
        return std::complex<double>(da, db);
    });
}

template <typename T>
void generate_random_vector(std::vector<T> &random_vector) {
    generate_random_array(random_vector.data(), random_vector.size());
}

template <typename T>
void generate_random_scalar(T &coefficient) {
    T lower_bound = 0;
    T upper_bound = 10000;
    std::random_device rd;
    std::mt19937 mersenne_engine(rd());
    std::uniform_real_distribution<T> dist(lower_bound, upper_bound);
    coefficient = dist(mersenne_engine);
}

template <typename T>
void generate_random_scalar(std::complex<T> &coefficient) {
    T lower_bound = 0;
    T upper_bound = 10000;
    std::random_device rd;
    std::mt19937 mersenne_engine(rd());
    std::uniform_real_distribution<T> dist(lower_bound, upper_bound);
    coefficient.real(dist(mersenne_engine));
    coefficient.imag(dist(mersenne_engine));
}

template <typename T>
void generate_random_matrix(Matrix<T> &matrix) {
    generate_random_array(matrix.data(), matrix.nb_cols() * matrix.nb_rows());
}

// template <>
// void generate_random_scalar(int &coefficient) {
//     int lower_bound = 0;
//     int upper_bound = 10000;
//     std::random_device rd;
//     std::mt19937 mersenne_engine(rd());
//     std::uniform_int_distribution<int> dist(lower_bound, upper_bound);
//     coefficient = dist(mersenne_engine);
// }

} // namespace htool

#endif
