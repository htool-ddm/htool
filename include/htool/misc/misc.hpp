#ifndef HTOOL_INFOS_HPP
#define HTOOL_INFOS_HPP

#include <algorithm>
#include <complex>
namespace htool {
template <class T>
struct underlying_type_spec {
    typedef T type;
};
template <class T>
struct underlying_type_spec<std::complex<T>> {
    typedef T type;
};
template <class T>
using underlying_type = typename underlying_type_spec<T>::type;

// Check if complex type
// https://stackoverflow.com/a/41438903/5913047
template <typename T>
struct is_complex_t : public std::false_type {};
template <typename T>
struct is_complex_t<std::complex<T>> : public std::true_type {};

template <typename T>
constexpr bool is_complex() { return is_complex_t<T>::value; }

// https://stackoverflow.com/a/63316255/5913047
template <typename T, typename std::enable_if<!is_complex_t<T>::value, int>::type = 0>
void conj_if_complex(T *, int) {}

template <typename T, typename std::enable_if<is_complex_t<T>::value, int>::type = 0>
void conj_if_complex(T *in, int size) {
    std::transform(in, in + size, in, [](const T &c) { return std::conj(c); });
}

template <typename T, typename std::enable_if<!is_complex_t<T>::value, int>::type = 0>
T conj_if_complex(T in) { return in; }

template <typename T, typename std::enable_if<is_complex_t<T>::value, int>::type = 0>
T conj_if_complex(T in) {
    return std::conj(in);
}
} // namespace htool
#endif
