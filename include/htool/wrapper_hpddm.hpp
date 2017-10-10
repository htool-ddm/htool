#ifndef WRAPPER_HPDDM_HPP
#define WRAPPER_HPDDM_HPP

#define HPDDM_NUMBERING 'C'
#include <HPDDM.hpp>
#include "hmatrix.hpp"
#include "matrix.hpp"

namespace htool{

// Friend functions
template< template<typename> class LowRankMatrix, typename T >
class HPDDMEmpty;

template< template<typename> class LowRankMatrix, typename T >
void solve(const HMatrix<LowRankMatrix,T>& HA, const T* const rhs, T* const x);

template< template<typename> class LowRankMatrix, typename T>
class HPDDMEmpty : public HPDDM::EmptyOperator<T> {
private:
  const HMatrix<LowRankMatrix,T>& HA;
  std::vector<T>* in_global;

public:
  // Constructor
  HPDDMEmpty(const HMatrix<LowRankMatrix,T>& A) : HPDDM::EmptyOperator<T>(A.get_local_cluster_size()), HA(A) {in_global = new std::vector<T> (HA.nb_cols());}
  ~HPDDMEmpty(){delete in_global;}

  void GMV(const T* const in, T* const out, const int& mu = 1) const {
    // All gather
    HA.local_to_global(in, in_global->data());

    //
    HA.mvprod_local(in_global->data(),out);
  }

  // Preconditioner
  template<bool = true>
  void apply(const T* const in, T* const out, const unsigned short& mu = 1, T* = nullptr, const unsigned short& = 0) const {
    std::copy_n(in, this->_n, out);
  }

  // solve
  friend void solve<LowRankMatrix,T>(const HMatrix<LowRankMatrix,T>& HA, const T* const rhs, T* const x);
};

template< template<typename> class LowRankMatrix, typename T>
void solve(const HMatrix<LowRankMatrix,T>& HA,const T* const rhs, T* const x){
  //
  int rankWorld = HA.get_rankworld();
  int sizeWorld = HA.get_sizeworld();
  int offset = HA.get_MasterOffset_t()[rankWorld].first;
  int size   = HA.get_MasterOffset_t()[rankWorld].second;

  //
  HPDDMEmpty<LowRankMatrix,T> A_HPDDM(HA);
  std::vector<T> rhs_perm(HA.nb_cols());
  std::vector<T> x_local(size);

  // Permutation
  HA.source_to_cluster_permutation(rhs,rhs_perm.data());

  // Solve
  HPDDM::IterativeMethod::solve(A_HPDDM, rhs_perm.data()+offset, x_local.data(), 1,HA.get_comm());

  // Local to global
  HA.local_to_global(x_local.data(),A_HPDDM.in_global->data());
	// Permutation
  HA.cluster_to_target_permutation(A_HPDDM.in_global->data(),x);

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
