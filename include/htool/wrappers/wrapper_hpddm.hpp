#ifndef HTOOL_WRAPPER_HPDDM_HPP
#define HTOOL_WRAPPER_HPDDM_HPP

#define HPDDM_NUMBERING 'F'
#define HPDDM_SCHWARZ 1
#define HPDDM_FETI 0
#define HPDDM_BDD 0
#define LAPACKSUB
#define DSUITESPARSE
#include <HPDDM.hpp>
#include "../types/hmatrix.hpp"
#include "../types/matrix.hpp"

namespace htool{

template<template<typename> class LowRankMatrix, typename T>
class ISolver;

template<template<typename> class LowRankMatrix, typename T>
class Schwarz;

template< template<typename> class LowRankMatrix, typename T>
class HPDDMDense : public HpDense<T> {
private:
  const HMatrix<LowRankMatrix,T>& HA;
  std::vector<T>* in_global,*buffer;


public:
  typedef  HpDense<T> super;

  HPDDMDense(const HMatrix<LowRankMatrix,T>& A):HA(A){
      in_global = new std::vector<T> ;
      buffer = new std::vector<T>;
  }
  ~HPDDMDense(){delete in_global;delete buffer;}


    void GMV(const T* const in, T* const out, const int& mu = 1) const {
        int local_size = HA.get_local_size();

        // Tranpose without overlap
        if (mu!=1){
            for (int i=0;i<mu;i++){
                for (int j=0;j<local_size;j++){
                    (*buffer)[i+j*mu]=in[i*this->getDof()+j];
                }
            }
        }

        // All gather
        if (mu==1){// C'est moche
            HA.mvprod_local(in,out,in_global->data(),mu);
        }
        else{
            HA.mvprod_local(buffer->data(),buffer->data()+local_size*mu,in_global->data(),mu);
        }



        // Tranpose
        if (mu!=1){
            for (int i=0;i<mu;i++){
                for (int j=0;j<local_size;j++){
                    out[i*this->getDof()+j]=(*buffer)[i+j*mu+local_size*mu];
                }
            }
        }
        this->scaledExchange(out, mu);
    }

  void exchange(T* const out, const int& mu = 1){
    this->template scaledExchange<true>(out, mu);
  }

  friend class ISolver<LowRankMatrix,T>;
  friend class Schwarz<LowRankMatrix,T>;


};


}
#endif
