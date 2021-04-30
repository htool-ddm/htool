#ifndef HTOOL_POINT_HPP
#define HTOOL_POINT_HPP

#include <algorithm>
#include <array>
#include <complex>
#include <iostream>
#include <iterator>
#include <numeric>

namespace htool {

typedef std::complex<double> Cplx;

typedef std::array<int, 4> N4;
typedef std::array<double, 2> R2;
typedef std::array<double, 3> R3;

template <typename T, std::size_t dim>
std::ostream &operator<<(std::ostream &out, const std::array<T, dim> &v) {
    if (!v.empty()) {
        out << '[';
        for (typename std::array<T, dim>::const_iterator i = v.begin(); i != v.end(); ++i)
            std::cout << *i << ',';
        out << "\b]";
    }
    return out;
}
template <typename T, std::size_t dim>
std::istream &operator>>(std::istream &is, std::array<T, dim> &a) {
    for (int j = 0; j < dim; j++) {
        is >> a[j];
    }
    return is;
}

template <typename T, std::size_t dim>
std::array<T, dim> operator+(const std::array<T, dim> &a, const std::array<T, dim> &b) {
    std::array<T, dim> result;
    std::transform(a.begin(), a.end(), b.begin(), result.begin(), std::plus<T>());

    return result;
}

template <typename T, std::size_t dim>
std::array<T, dim> operator-(const std::array<T, dim> &a, const std::array<T, dim> &b) {
    std::array<T, dim> result;
    std::transform(a.begin(), a.end(), b.begin(), result.begin(), std::minus<T>());

    return result;
}

template <typename T, std::size_t dim>
std::array<T, dim> operator*(T value, const std::array<T, dim> &a) {
    std::array<T, dim> result;
    std::transform(a.begin(), a.end(), result.begin(), [value](const T &c) { return c * value; });

    return result;
}

template <typename T, std::size_t dim>
std::array<T, dim> operator*(const std::array<T, dim> &b, T value) {
    return value * b;
}

template <typename T, std::size_t dim>
void operator+=(std::array<T, dim> &a, const std::array<T, dim> &b) {
    std::transform(a.begin(), a.end(), b.begin(), a.begin(), std::plus<T>());
}

template <typename T, std::size_t dim>
void operator*=(std::array<T, dim> &a, const T &value) {
    std::transform(a.begin(), a.end(), a.begin(), [value](T &c) { return c * value; });
}

template <typename T, std::size_t dim>
void operator/=(std::array<T, dim> &a, const T &value) {
    std::transform(a.begin(), a.end(), a.begin(), [value](T &c) { return c / value; });
}

template <typename T, std::size_t dim>
T dprod(const std::array<T, dim> &a, const std::array<T, dim> &b) {
    return std::inner_product(a.begin(), a.end(), b.begin(), T());
}
template <typename T, std::size_t dim>
std::complex<T> dprod(const std::array<std::complex<T>, dim> &a, const std::array<std::complex<T>, dim> &b) {
    return std::inner_product(a.begin(), a.end(), b.begin(), std::complex<T>(), std::plus<std::complex<T>>(), [](std::complex<T> u, std::complex<T> v) { return u * std::conj<T>(v); });
}

template <typename T, std::size_t dim>
T operator,(const std::array<T, dim> &a, const std::array<T, dim> &b) {
    return std::inner_product(a.begin(), a.end(), b.begin(), T());
}
template <typename T, std::size_t dim>
std::complex<T> operator,(const std::array<std::complex<T>, dim> &a, const std::array<std::complex<T>, dim> &b) {
    return std::inner_product(a.begin(), a.end(), b.begin(), std::complex<T>(), std::plus<std::complex<T>>(), [](std::complex<T> u, std::complex<T> v) { return u * std::conj<T>(v); });
}

template <typename T, std::size_t dim>
T norm2(const std::array<T, dim> &u) { return std::sqrt(std::abs(dprod(u, u))); }

template <typename T, std::size_t dim>
T norm2(const std::array<std::complex<T>, dim> &u) { return std::sqrt(std::abs(dprod(u, u))); }

inline R3 operator^(const R3 &N, const R3 &P) {
    R3 res;
    res[0] = N[1] * P[2] - N[2] * P[1];
    res[1] = N[2] * P[0] - N[0] * P[2];
    res[2] = N[0] * P[1] - N[1] * P[0];
    return res;
}

} // namespace htool

#endif
