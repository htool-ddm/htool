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

// // Friend functions
// template< template<typename> class LowRankMatrix, typename T >
// class HPDDMEmpty;
//
// template< template<typename> class LowRankMatrix, typename T >
// void solve(const HMatrix<LowRankMatrix,T>& HA, const T* const rhs, T* const x);
//
// template< template<typename> class LowRankMatrix, typename T>
// class HPDDMEmpty : public HPDDM::EmptyOperator<T> {
// private:
//   const HMatrix<LowRankMatrix,T>& HA;
//   std::vector<T>* in_global;
//
// public:
//   // Constructor
//   HPDDMEmpty(const HMatrix<LowRankMatrix,T>& A) : HPDDM::EmptyOperator<T>(A.get_local_size()), HA(A) {in_global = new std::vector<T> (HA.nb_cols());}
//   ~HPDDMEmpty(){delete in_global;}
//
//   void GMV(const T* const in, T* const out, const int& mu = 1) const {
//     // All gather
//     HA.mvprod_local(in,out,in_global->data(),mu);
//
//     // std::copy_n(in, this->_n, out);
//   }
//
//   // Preconditioner
//   template<bool = true>
//   void apply(const T* const in, T* const out, const unsigned short& mu = 1, T* = nullptr, const unsigned short& = 0) const {
//     std::copy_n(in, mu*this->_n, out);
//   }
//
//   // solve
//   void solve(const T* const rhs, T* const x, const int& mu=1){
//     //
//     int rankWorld = HA.get_rankworld();
//     int sizeWorld = HA.get_sizeworld();
//     int offset = HA.get_local_offset();
//     int size   = HA.get_local_size();
//     int nb_cols = HA.nb_cols();
//     double time = MPI_Wtime();
//
//     //
//     std::vector<T> rhs_perm(nb_cols);
//     std::vector<T> x_local(size*mu);
//     std::vector<T> local_rhs(size*mu,0);
//     in_global->resize(nb_cols*(mu==1 ? 1 : 2*mu),2); // used for rearranging rhss after allgather in local_to_global
//
//     for (int i=0;i<mu;i++){
//       // Permutation
//       HA.source_to_cluster_permutation(rhs+i*nb_cols,rhs_perm.data());
//
//       std::copy_n(rhs_perm.begin()+offset,size,local_rhs.begin()+i*size);
//     }
//
//     // Solve
//     HPDDM::IterativeMethod::solve(*this, local_rhs.data(), x_local.data(), mu,HA.get_comm());
//
//     // Local to global
//     HA.local_to_global(x_local.data(),in_global->data(),mu);
//
//     for (int i=0;i<mu;i++){
//       // Permutation
//       HA.cluster_to_target_permutation(in_global->data()+i*nb_cols,x+i*nb_cols);
//     }
//
//
//
//   }
// };

// template< template<typename> class LowRankMatrix, typename T>
// void solve(HMatrix<LowRankMatrix,T>& HA,const T* const rhs, T* const x){
//   //
//   int rankWorld = HA.get_rankworld();
//   int sizeWorld = HA.get_sizeworld();
//   int offset = HA.get_local_offset();
//   int size   = HA.get_local_size();
//
//   //
//   HPDDMEmpty<LowRankMatrix,T> A_HPDDM(HA);
//   std::vector<T> rhs_perm(HA.nb_cols());
//   std::vector<T> x_local(size);
//
//   // Permutation
//   HA.source_to_cluster_permutation(rhs,rhs_perm.data());
//
//   // Solve
//   // std::cout << rhs_perm << std::endl;
//   // std::cout << x_local << std::endl;
//   // std::cout << x_local.size()<<" "<<rhs_perm.size()<<std::endl;
//   // std::cout << offset << std::endl;
//   HPDDM::IterativeMethod::solve(A_HPDDM, rhs_perm.data()+offset, x_local.data(), 1,HA.get_comm());
//
//   // Local to global
//   HA.local_to_global(x_local.data(),A_HPDDM.in_global->data());
// 	// Permutation
//   HA.cluster_to_target_permutation(A_HPDDM.in_global->data(),x);
//
// }

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

  // void GMV(const T* const in, T* const out, const int& mu = 1) const {
  //     int n_inside = HA.get_local_size();
  //     std::vector<T> buffer(n_inside*mu*2);
  //
  //     if (mu>1){
  //         for (int i =0;i<mu;i++){
  //             std::copy_n(in+i*this->getDof(),n_inside,buffer.data()+i*n_inside);
  //         }
  //
  //     }
  //   // All gather
  //   if (mu==1){// C'est moche
  //       HA.mvprod_local(in,out,in_global->data(),mu);
  //   }
  //   else{
  //       HA.mvprod_local(buffer.data(),buffer.data()+n_inside*mu,in_global->data(),mu);
  //   }
  //   if (mu>1){
  //       for (int i =0;i<mu;i++){
  //           std::copy_n(buffer.data()+n_inside*mu+i*n_inside,n_inside,out+i*this->getDof());
  //       }
  //
  //   }
  //   this->scaledExchange(out, mu);
  // }

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

  friend class Schwarz<LowRankMatrix,T>;


};


}
#endif
