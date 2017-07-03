#ifndef HPDDM_CALL_H
#define HPDDM_CALL_H

#define HPDDM_NUMBERING 'C'
#include <HPDDM.hpp>
#include "hmatrix.hpp"
namespace htool{

template< template<typename> class LowRankMatrix, typename T>
struct HPDDMOperator : HPDDM::EmptyOperator<T> {
  HMatrix<LowRankMatrix,T>& HA;
  HPDDMOperator(HMatrix<LowRankMatrix,T>& A) : HPDDM::EmptyOperator<T>(A.nb_rows()), HA(A) { }
  void GMV(const T* const in, T* const out, const int& mu = 1) const {
    // HA.mvprod(in,out);
    std::copy_n(in, this->_n, out);
    // std::cout << "ok" << std::endl;
    // for (int i =0;i<this->_n;i++){
    //   std::cout << in[i]<<" ";
    // }
    // std::cout <<std::endl;
  }
  template<bool = true>
  void apply(const T* const in, T* const out, const unsigned short& mu = 1, T* = nullptr, const unsigned short& = 0) const {
    // for(int i = 0; i < _n; ++i) {
    //   // #if 1
    //   // out[i] = in[i] / _A(i, i);
    //   // #else
    //   out[i] = in[i];
    //   // #endif
    // }
    std::copy_n(in, this->_n, out);
  }
};


}
#endif
