#ifndef HTOOL_VECTOR_HPP
#define HTOOL_VECTOR_HPP

#include <algorithm>
#include <cassert>
#include <complex>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

namespace htool {
//================================//
//      DECLARATIONS DE TYPE      //
//================================//
typedef std::pair<int, int> Int2;

//================================//
//            VECTEUR             //
//================================//
template <typename T>
std::ostream &operator<<(std::ostream &out, const std::vector<T> &v) {
    if (!v.empty()) {
        out << '[';
        for (typename std::vector<T>::const_iterator i = v.begin(); i != v.end(); ++i)
            std::cout << *i << ',';
        out << "\b]";
    }
    return out;
}

template <typename T>
std::vector<T> operator+(const std::vector<T> &a, const std::vector<T> &b) {
    assert(a.size() == b.size());
    std::vector<T> result(a.size(), 0);
    std::transform(a.begin(), a.end(), b.begin(), result.begin(), std::plus<T>());

    return result;
}

template <typename T>
std::vector<T> plus(const std::vector<T> &a, const std::vector<T> &b) {

    return a + b;
}

template <typename T>
std::vector<T> operator-(const std::vector<T> &a, const std::vector<T> &b) {
    assert(a.size() == b.size());
    std::vector<T> result(a.size(), 0);
    std::transform(a.begin(), a.end(), b.begin(), result.begin(), std::minus<T>());

    return result;
}

template <typename T>
std::vector<T> minus(const std::vector<T> &a, const std::vector<T> &b) {

    return a - b;
}

// template<typename T>
// std::vector<T> mult(V value,const std::vector<T>& a)
// {
// 	std::vector<T> result(a.size(),0);
// 	std::transform (a.begin(), a.end(), result.begin(), std::bind1st(std::multiplies<T>(),value));
//
// 	return result;
// }
//
// template<typename T, typename V>
// std::vector<T> mult(const std::vector<T>& b,V value)
// {
// 	return value*b;
// }

template <typename T, typename V>
std::vector<T> operator/(const std::vector<T> &a, V value) {
    std::vector<T> result(a.size(), 0);
    std::transform(a.begin(), a.end(), result.begin(), [value](const T &c) { return c / value; });

    return result;
}

template <typename T>
T dprod(const std::vector<T> &a, const std::vector<T> &b) {
    return std::inner_product(a.begin(), a.end(), b.begin(), T(0));
}
template <typename T>
std::complex<T> dprod(const std::vector<std::complex<T>> &a, const std::vector<std::complex<T>> &b) {
    return std::inner_product(a.begin(), a.end(), b.begin(), std::complex<T>(0), std::plus<std::complex<T>>(), [](std::complex<T> u, std::complex<T> v) { return u * std::conj<T>(v); });
}

template <typename T>
double norm2(const std::vector<T> &u) { return std::sqrt(std::abs(dprod(u, u))); }

template <typename T>
T max(const std::vector<T> &u) {
    return *std::max_element(u.begin(), u.end(), [](T a, T b) { return std::abs(a) < std::abs(b); });
}

template <typename T>
T min(const std::vector<T> &u) {
    return *std::min_element(u.begin(), u.end(), [](T a, T b) { return std::abs(a) < std::abs(b); });
}

template <typename T>
int argmax(const std::vector<T> &u) {
    return std::max_element(u.begin(), u.end(), [](T a, T b) { return std::abs(a) < std::abs(b); }) - u.begin();
}

template <typename T, typename V>
void operator*=(std::vector<T> &a, const V &value) {
    std::transform(a.begin(), a.end(), a.begin(), [value](T &c) { return c * value; });
}

template <typename T, typename V>
void operator/=(std::vector<T> &a, const V &value) {
    std::transform(a.begin(), a.end(), a.begin(), [value](T &c) { return c / value; });
}

template <typename T>
T mean(const std::vector<T> &u) {
    return std::accumulate(u.begin(), u.end(), T(0)) / T(u.size());
}

template <typename T>
int vector_to_bytes(const std::vector<T> vect, const std::string &file) {
    std::ofstream out(file, std::ios::out | std::ios::binary | std::ios::trunc);

    if (!out) {
        std::cout << "Cannot open file: " << file << std::endl; // LCOV_EXCL_LINE
        return 1;                                               // LCOV_EXCL_LINE
    }
    int size = vect.size();
    out.write((char *)(&size), sizeof(int));
    out.write((char *)&(vect[0]), size * sizeof(T));

    out.close();
    return 0;
}

template <typename T>
int bytes_to_vector(std::vector<T> &vect, const std::string &file) {

    std::ifstream in(file, std::ios::in | std::ios::binary);

    if (!in) {
        std::cout << "Cannot open file: " << file << std::endl; // LCOV_EXCL_LINE
        return 1;                                               // LCOV_EXCL_LINE
    }

    int size = 0;
    in.read((char *)(&size), sizeof(int));
    vect.resize(size);
    in.read((char *)&(vect[0]), size * sizeof(T));

    in.close();
    return 0;
}

// To be used with dlmread
template <typename T>
int matlab_save(std::vector<T> vector, const std::string &file) {
    std::ofstream out(file);
    out << std::setprecision(18);
    if (!out) {
        std::cout << "Cannot open file: " << file << std::endl;
        return 1;
    }

    // out<<rows<<" "<<cols<<std::endl;
    for (int i = 0; i < vector.size(); i++) {
        out << std::real(vector[i]);
        if (std::imag(vector[i]) < 0) {
            out << std::imag(vector[i]) << "i\t";
        } else if (std::imag(vector[i]) == 0) {
            out << "+" << 0 << "i\t";
        } else {
            out << "+" << std::imag(vector[i]) << "i\t";
        }
        out << std::endl;
    }
    out.close();
    return 0;
}

//================================//
//      CLASSE SUBVECTOR          //
//================================//

template <typename T>
class SubVec {

  private:
    std::vector<T> &U;
    const std::vector<int> &I;
    const int size;

  public:
    SubVec(std::vector<T> &&U0, const std::vector<int> &I0) : U(U0), I(I0), size(I0.size()) {}
    SubVec(const SubVec &); // Pas de constructeur par recopie

    T &operator[](const int &k) { return U[I[k]]; }
    const T &operator[](const int &k) const { return U[I[k]]; }

    template <typename RhsType>
    T operator,(const RhsType &rhs) const {
        T lhs = 0.;
        for (int k = 0; k < size; k++) {
            lhs += U[I[k]] * rhs[k];
        }
        return lhs;
    }

    int get_size() const { return this->size; }

    friend std::ostream &operator<<(std::ostream &os, const SubVec &u) {
        for (int j = 0; j < u.size; j++) {
            os << u[j] << "\t";
        }
        return os;
    }
};

} // namespace htool

#endif
