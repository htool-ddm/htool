#ifndef POINT_HPP
#define POINT_HPP

#include <iostream>
#include <cmath>

namespace htool {

typedef double                   Real;
typedef std::complex<Real>            Cplx;

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

  friend v_t operator,(const a_t& l, const a_t& r){
    v_t v=0; for(int j=0; j<dim; j++){v+=l[j]*r[j];} return v;}
  
};

template <int dim>
Real norm(const array_<dim,Real>& u){
  Real r = 0.; for(int j=0; j<dim; j++){r+=u[j]*u[j];}
  return sqrt(r);}

template <int dim>
Real norm(const array_<dim,Cplx>& u){
  Real r = 0.; for(int j=0; j<dim; j++){r+=abs(u[j])*abs(u[j]);}
  return sqrt(r);}

typedef array_<2,int>  N2;
typedef array_<3,int>  N3;
typedef array_<4,int>  N4;
typedef array_<2,Real> R2;
typedef array_<3,Real> R3;
typedef array_<2,Cplx> C2;
typedef array_<3,Cplx> C3;

typedef array_<2,R2> R2x2;
typedef array_<2,R3> R3x2;
typedef array_<3,R2> R2x3;
typedef array_<3,R3> R3x3;
typedef array_<4,R2> R2x4;
typedef array_<4,R3> R3x4;
}

#endif
