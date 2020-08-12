#ifndef HTOOL_INFOS_HPP
#define HTOOL_INFOS_HPP

#include <complex>



// Check if complex type
// https://stackoverflow.com/a/41438903/5913047
template<typename T>
struct is_complex_t : public std::false_type {};
template<typename T>
struct is_complex_t<std::complex<T>> : public std::true_type {};


template<typename T>
constexpr bool is_complex() { return is_complex_t<T>::value; }


// https://stackoverflow.com/a/63316255/5913047
template<typename T, typename std::enable_if<!is_complex_t<T>::value, int>::type = 0>
void conj_if_complex(T* in, int size){}

template<typename T,  typename std::enable_if<is_complex_t<T>::value, int>::type = 0>
void conj_if_complex(T* in, int size){
    if (is_complex<T>()){
        std::transform(in,in+size,in,[](const T& c){ return std::conj(c); });
    }
}


#endif