#ifndef HPDDM_CALL_H
#define HPDDM_CALL_H

#define HPDDM_NUMBERING 'C'
#include <HPDDM.hpp>
#include "hmatrix.hpp"
#include "matrix.hpp"
namespace htool{

template< template<typename> class LowRankMatrix, typename T>
struct HPDDMOperator : HPDDM::EmptyOperator<T> {
  HMatrix<LowRankMatrix,T>& HA;
  Preconditioner& P;
  HPDDMOperator(HMatrix<LowRankMatrix,T>& A,Preconditioner& P0) : HPDDM::EmptyOperator<T>(A.get_local_size_cluster()), HA(A), P(P0) {}
  void GMV(const T* const in, T* const out, const int& mu = 1) const {
    HA.mvprod_local(in,out);
  }
  template<bool = true>
  void apply(const T* const in, T* const out, const unsigned short& mu = 1, T* = nullptr, const unsigned short& = 0) const {
    // std::copy_n(in, this->_n, out);
    P.apply(in,out);
//     for(int i = 0; i < this->_n; ++i) {
//   #if 1
//   out[i] = in[i] / P.get_coef(i, i);
//   #else
//   out[i] = in[i];
//   #endif
// }
  }
};


}
#endif
