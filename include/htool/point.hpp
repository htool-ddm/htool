#ifndef POINT_HPP
#define POINT_HPP

#include <complex>

namespace htool {

typedef std::complex<double>       Cplx;

template<int dim, typename v_t>
class array_{

public:
  typedef array_<dim,v_t> a_t;

private:
  v_t v_[dim];

public:

  array_<dim,v_t>(const v_t& v0 = v_t()){
    for(int j=0; j<dim; j++){v_[j]=v0;}}

  array_<dim,v_t>(const a_t& a){
    for(int j=0; j<dim; j++){v_[j]=a.v_[j];}}

  void operator=(const v_t& v0){
    for(int j=0; j<dim; j++){v_[j]=v0;}}

  void operator=(const a_t& a){
    for(int j=0; j<dim; j++){v_[j]=a.v_[j];}}

  void operator+=(const v_t& v0){
    for(int j=0; j<dim; j++){v_[j]+=v0;}}

  void operator*=(const v_t& v0){
    for(int j=0; j<dim; j++){v_[j]*=v0;}}

  void operator/=(const v_t& v0){
    for(int j=0; j<dim; j++){v_[j]/=v0;}}

  void operator+=(const a_t& a){
    for(int j=0; j<dim; j++){v_[j]+=a.v_[j];}}

  v_t& operator[](const int& j){return v_[j];}

  const v_t& operator[](const int& j) const {return v_[j];}

  friend std::ostream& operator<<(std::ostream& os, const a_t& a){
    for(int j=0; j<dim; j++){ os << a[j] << "\t";} return os;}

  friend std::istream& operator>>(std::istream& is, a_t& a){
    for(int j=0; j<dim; j++){ is >> a[j];} return is;}

  friend a_t operator+(const a_t& l, const a_t& r){
    a_t a; for(int j=0; j<dim; j++){a[j]=l[j]+r[j];} return a;}

  friend a_t operator-(const a_t& l, const a_t& r){
    a_t a; for(int j=0; j<dim; j++){a[j]=l[j]-r[j];} return a;}

  friend a_t operator*(const v_t& l, const a_t& r){
    a_t a; for(int j=0; j<dim; j++){a[j]=l*r[j];} return a;}

  friend a_t operator*(const a_t& r, const v_t& l){
    a_t a; for(int j=0; j<dim; j++){a[j]=l*r[j];} return a;}

  friend a_t operator/(const a_t& r, const v_t& l){
    a_t a; for(int j=0; j<dim; j++){a[j]=r[j]/l;} return a;}

  friend v_t operator,(const a_t& l, const a_t& r){
    v_t v=0; for(int j=0; j<dim; j++){v+=l[j]*r[j];} return v;}

};

template <int dim>
double norm(const array_<dim,double>& u){
  double r = 0.; for(int j=0; j<dim; j++){r+=u[j]*u[j];}
  return sqrt(r);}

template <int dim>
double norm(const array_<dim,Cplx>& u){
  double r = 0.; for(int j=0; j<dim; j++){r+=abs(u[j])*abs(u[j]);}
  return sqrt(r);}

typedef array_<2,int>  N2;
typedef array_<3,int>  N3;
typedef array_<4,int>  N4;
typedef array_<2,double> R2;
typedef array_<3,double> R3;
typedef array_<2,Cplx> C2;
typedef array_<3,Cplx> C3;

typedef array_<2,R2> R2x2;
typedef array_<2,R3> R3x2;
typedef array_<3,R2> R2x3;
typedef array_<3,R3> R3x3;
typedef array_<4,R2> R2x4;
typedef array_<4,R3> R3x4;

R3  operator^(const R3 &N, const R3 &P) {R3 res; res[0] = N[1]*P[2]-N[2]*P[1] ; res[1] = N[2]*P[0]-N[0]*P[2]; res[2] = N[0]*P[1]-N[1]*P[0]; return res;}
}

#endif
