#ifndef HTOOL_POINT_HPP
#define HTOOL_POINT_HPP

#include <complex>
#include <array>
#include <numeric>
#include <iterator>
#include <iostream>
#include <algorithm>

namespace htool {

typedef std::complex<double>       Cplx;

typedef std::array<int,2>  N2;
typedef std::array<int,3>  N3;
typedef std::array<int,4>  N4;
typedef std::array<double,2> R2;
typedef std::array<double,3> R3;
typedef std::array<std::complex<double>,2> C2;
typedef std::array<std::complex<double>,3> C3;

template <typename T,std::size_t dim>
std::ostream& operator<< (std::ostream& out, const std::array<T,dim>& v) {
  if ( !v.empty() ) {
    out << '[';
    for (typename std::array<T,dim>::const_iterator i = v.begin(); i != v.end(); ++i)
    std::cout << *i << ',';
    out << "\b]";
  }
  return out;
}
template<typename T, std::size_t dim>
std::istream& operator>>(std::istream& is, std::array<T,dim>& a){
    for(int j=0; j<dim; j++){ is >> a[j];}
  return is;
}

template<typename T,std::size_t dim>
std::array<T,dim> operator+(const std::array<T,dim>& a,const std::array<T,dim>& b)
{
	std::array<T,dim> result;
	std::transform (a.begin(), a.end(), b.begin(), result.begin(), std::plus<T>());

	return result;
}

template<typename T,std::size_t dim>
std::array<T,dim> operator-(const std::array<T,dim>& a,const std::array<T,dim>& b)
{
	std::array<T,dim> result;
	std::transform (a.begin(), a.end(), b.begin(), result.begin(), std::minus<T>());

	return result;
}

template<typename T,std::size_t dim>
std::array<T,dim> operator*(T value,const std::array<T,dim>& a)
{
	std::array<T,dim> result;
	std::transform (a.begin(), a.end(), result.begin(), [value](const T& c){return c*value;});

	return result;
}

template<typename T,std::size_t dim>
std::array<T,dim> operator*(const std::array<T,dim>& b,T value)
{
	return value*b;
}

template <typename T, std::size_t dim>
void operator+=(std::array<T,dim> &a, const std::array<T,dim> &b) {
    std::transform (a.begin(), a.end(), b.begin(),a.begin(), std::plus<T>());
}

template <typename T, std::size_t dim>
void operator*=(std::array<T,dim> &a, const T &value) {
    std::transform (a.begin(), a.end() ,a.begin(), [value](T& c){return c*value;});
}

template <typename T, std::size_t dim>
void operator/=(std::array<T,dim> &a, const T &value) {
    std::transform (a.begin(), a.end() ,a.begin(), [value](T& c){return c/value;});
}

template <typename T, std::size_t dim>
T dprod(const std::array<T,dim>& a,const std::array<T,dim>& b){
	return std::inner_product(a.begin(),a.end(),b.begin(),T());
}
template <typename T, std::size_t dim>
std::complex<T> dprod(const std::array<std::complex<T>,dim >& a,const std::array<std::complex<T> , dim>& b){
	return std::inner_product(a.begin(),a.end(),b.begin(),std::complex<T>(),std::plus<std::complex<T> >(), [](std::complex<T>u,std::complex<T>v){return u*std::conj<T>(v);});
}

template <typename T, std::size_t dim>
T operator,(const std::array<T,dim>& a,const std::array<T,dim>& b){
	return std::inner_product(a.begin(),a.end(),b.begin(),T());
}
template <typename T, std::size_t dim>
std::complex<T> operator,(const std::array<std::complex<T>,dim >& a,const std::array<std::complex<T> , dim>& b){
	return std::inner_product(a.begin(),a.end(),b.begin(),std::complex<T>(),std::plus<std::complex<T> >(), [](std::complex<T>u,std::complex<T>v){return u*std::conj<T>(v);});
}

template <typename T, std::size_t dim>
T norm2(const std::array<T,dim>& u){return std::sqrt(std::abs(dprod(u,u)));}

template <typename T, std::size_t dim>
T norm2(const std::array<std::complex<T>,dim >& u){return std::sqrt(std::abs(dprod(u,u)));}

R3 operator^(const R3 &N, const R3 &P) {R3 res; res[0] = N[1]*P[2]-N[2]*P[1] ; res[1] = N[2]*P[0]-N[0]*P[2]; res[2] = N[0]*P[1]-N[1]*P[0]; return res;}

// template<typename T, std::size_t dim>


// template<typename T,int dim>
// class Point{
//
// private:
//     std::vector<T> v;
//
// public:
//     // LowRankMatrix(const LowRankMatrix&) = default; // copy constructor
//     // LowRankMatrix& operator=(const LowRankMatrix&) = default; // copy assignement operator
//     // LowRankMatrix(LowRankMatrix&&) = default; // move constructor
//     // LowRankMatrix& operator=(LowRankMatrix&&) = default; // move assignement operator
//
//   Point():v(3,0){}
//   Point(std::vector<T> vect){assert(vect.size()==dim);v=vect;}
//   Point(const Point&) = default;   // copy constructor
//   Point& operator=(const Point&) = default;   // copy assignement operator
//   Point(Point&&) = default;    // move constructor
//   Point& operator=(Point&&) = default; // move assignement operator
// //
// //   array_<dim,v_t>(const v_t& v0 = v_t()){
// //     for(int j=0; j<dim; j++){v_[j]=v0;}}
// //
// //   array_<dim,v_t>(const a_t& a){
// //     for(int j=0; j<dim; j++){v_[j]=a.v_[j];}}
// //
// //   void operator=(const v_t& v0){
// //     for(int j=0; j<dim; j++){v_[j]=v0;}}
// //
// //   void operator=(const a_t& a){
// //     for(int j=0; j<dim; j++){v_[j]=a.v_[j];}}
// //
// //   void operator+=(const v_t& v0){
// //     for(int j=0; j<dim; j++){v_[j]+=v0;}}
// //
// //   void operator*=(const v_t& v0){
// //     for(int j=0; j<dim; j++){v_[j]*=v0;}}
// //
// //   void operator/=(const v_t& v0){
// //     for(int j=0; j<dim; j++){v_[j]/=v0;}}
// //
// //   void operator+=(const a_t& a){
// //     for(int j=0; j<dim; j++){v_[j]+=a.v_[j];}}
// //
// //   v_t& operator[](const int& j){return v_[j];}
// //
// //   const v_t& operator[](const int& j) const {return v_[j];}
// //
//   friend std::ostream& operator<<(std::ostream& out, const Point& p){
//     if ( !p.v.empty() ) {
//       out << '[';
//       std::copy (p.v.begin(), p.v.end(), std::ostream_iterator<T>(out, ", "));
//       out << "\b\b]";
//     }
//     return out;
//   }
// //
// //   friend std::istream& operator>>(std::istream& is, a_t& a){
// //     for(int j=0; j<dim; j++){ is >> a[j];} return is;}
// //
// //   friend a_t operator+(const a_t& l, const a_t& r){
// //     a_t a; for(int j=0; j<dim; j++){a[j]=l[j]+r[j];} return a;}
// //
// //   friend a_t operator-(const a_t& l, const a_t& r){
// //     a_t a; for(int j=0; j<dim; j++){a[j]=l[j]-r[j];} return a;}
// //
// //   friend a_t operator*(const v_t& l, const a_t& r){
// //     a_t a; for(int j=0; j<dim; j++){a[j]=l*r[j];} return a;}
// //
// //   friend a_t operator*(const a_t& r, const v_t& l){
// //     a_t a; for(int j=0; j<dim; j++){a[j]=l*r[j];} return a;}
// //
// //   friend a_t operator/(const a_t& r, const v_t& l){
// //     a_t a; for(int j=0; j<dim; j++){a[j]=r[j]/l;} return a;}
// //
// //   friend v_t operator,(const a_t& l, const a_t& r){
// //     v_t v=0; for(int j=0; j<dim; j++){v+=l[j]*r[j];} return v;}
// //
// };
//
// template <int dim>
// double norm(const array_<dim,double>& u){
//   double r = 0.; for(int j=0; j<dim; j++){r+=u[j]*u[j];}
//   return sqrt(r);}
//
// template <int dim>
// double norm(const array_<dim,Cplx>& u){
//   double r = 0.; for(int j=0; j<dim; j++){r+=abs(u[j])*abs(u[j]);}
//   return sqrt(r);}
//

//
// typedef array_<2,R2> R2x2;
// typedef array_<2,R3> R3x2;
// typedef array_<3,R2> R2x3;
// typedef array_<3,R3> R3x3;
// typedef array_<4,R2> R2x4;
// typedef array_<4,R3> R3x4;
//
// R3  operator^(const R3 &N, const R3 &P) {R3 res; res[0] = N[1]*P[2]-N[2]*P[1] ; res[1] = N[2]*P[0]-N[0]*P[2]; res[2] = N[0]*P[1]-N[1]*P[0]; return res;}
}

#endif
