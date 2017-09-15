#ifndef WRAPPER_HPDDM_HPP
#define WRAPPER_HPDDM_HPP

#define HPDDM_NUMBERING 'C'
#include <HPDDM.hpp>
#include "hmatrix.hpp"
#include "matrix.hpp"

namespace htool{

template< template<typename> class LowRankMatrix, typename T>
class HPDDMEmpty : public HPDDM::EmptyOperator<T> {
  const HMatrix<LowRankMatrix,T>& HA;
public:
  HPDDMEmpty(const HMatrix<LowRankMatrix,T>& A) : HPDDM::EmptyOperator<T>(A.get_local_size_cluster()), HA(A) {}
  void GMV(const T* const in, T* const out, const int& mu = 1) const {
    HA.mvprod_local(in,out);
  }
  template<bool = true>
  void apply(const T* const in, T* const out, const unsigned short& mu = 1, T* = nullptr, const unsigned short& = 0) const {
    std::copy_n(in, this->_n, out);
  }
};

template< template<typename> class LowRankMatrix, typename T>
void solve(const HMatrix<LowRankMatrix,T>& HA, std::vector<T>& x0,  const std::vector<T> &  rhs0){
  HPDDMEmpty<LowRankMatrix,T> A_HPDDM(HA);
  
  HPDDM::IterativeMethod::solve(A_HPDDM, rhs0.data(), x0.data(), 1,HA.get_comm());
}




template< template<typename> class LowRankMatrix, typename T>
class HPDDMDense : public HpDense<T> {
private:
  const HMatrix<LowRankMatrix,T>& HA;

public:

  HPDDMDense(const HMatrix<LowRankMatrix,T>& A):HA(A){}

  void GMV(const T* const in, T* const out, const int& mu = 1) const {
    std::cout << "àk" << std::endl;
    HA.mvprod_local(in,out);
    this->super::scaledExchange(out, mu);

  }



};


}
#endif
