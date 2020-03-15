#ifndef HTOOL_MULTI_LRMAT_HPP
#define HTOOL_MULTI_LRMAT_HPP

#include <vector>
#include "../clustering/cluster.hpp"
#include "../types/multimatrix.hpp"
#include "../lrmat/lrmat.hpp"
namespace htool{

template< template<typename> class LowRankMatrix, typename T >
class MultiLowRankMatrix: public Parametres{

protected:
    // Data member
    int rank, nr, nc, nm;
    // Matrix<T>  U,V;
    std::vector<LowRankMatrix<T>> LowRankMatrices;
    std::vector<int> ir;
    std::vector<int> ic;
    int offset_i;
    int offset_j;

public:

    // Constructors
    MultiLowRankMatrix() = delete;
    MultiLowRankMatrix(const std::vector<int>& ir0, const std::vector<int>& ic0, int nm0, int rank0=-1):rank(rank0), nr(ir0.size()), nc(ic0.size()), nm(nm0), ir(ir0), ic(ic0), offset_i(0), offset_j(0){
        for (int l=0;l<nm;l++){
            LowRankMatrices.emplace_back(ir,ic,rank0);
        }
    }
    MultiLowRankMatrix(const std::vector<int>& ir0, const std::vector<int>& ic0, int nm0, int offset_i0, int offset_j0, int rank0=-1):rank(rank0), nr(ir0.size()), nc(ic0.size()), nm(nm0), ir(ir0),ic(ic0),offset_i(offset_i0), offset_j(offset_j0){
        for (int l=0;l<nm;l++){
            LowRankMatrices.emplace_back(ir,ic,offset_i,offset_j,rank0);
        }
    }

    // VIrtual function
    virtual void build(const MultiIMatrix<T>& A, const Cluster& t, const std::vector<R3>& xt,const std::vector<int>& tabt, const Cluster& s, const std::vector<R3>& xs, const std::vector<int>& tabs) = 0;

    // Getters
    int nb_rows()   const {return this->nr;}
    int nb_cols()   const {return this->nc;}
    int nb_lrmats() const {return this->nm;}
    // int rank_of() const {return this->rank;}
    std::vector<int> get_ir() const {return this->ir;}
    std::vector<int> get_ic() const {return this->ic;}
    int get_offset_i() const {return this->offset_i;}
    int get_offset_j() const {return this->offset_j;}

    LowRankMatrix<T>&  operator[](int j){return LowRankMatrices[j];}; 
    const LowRankMatrix<T>&  operator[](int j) const {return LowRankMatrices[j];}; 


};

template< template<typename> class LowRankMatrix, typename T >
double Frobenius_absolute_error(const MultiLowRankMatrix<LowRankMatrix,T>& lrmat, const MultiIMatrix<T>& ref, int l, int reqrank=-1){
  assert(reqrank<=lrmat[l].rank_of());
  if (reqrank==-1){
    reqrank=lrmat[l].rank_of();
  }
  T err = 0;
  std::vector<int> ir = lrmat.get_ir();
  std::vector<int> ic = lrmat.get_ic();
  for (int j=0;j<lrmat.nb_rows();j++){
    for (int k=0;k<lrmat.nb_cols();k++){
      T aux=ref.get_coefs(ir[j],ic[k])[l];
      for (int r=0;r<reqrank;r++){
        aux = aux - lrmat[l].get_U(j,r) * lrmat[l].get_V(r,k);
      }
      err+=std::pow(std::abs(aux),2);
    }
  }
  return std::sqrt(err);
}

template< template<typename> class LowRankMatrix, typename T >
double Frobenius_absolute_error(const MultiLowRankMatrix<LowRankMatrix,std::complex<T>>& lrmat, const MultiIMatrix<std::complex<T>>& ref, int l, int reqrank=-1){
  assert(reqrank<=lrmat[l].rank_of());
  if (reqrank==-1){
    reqrank=lrmat.rank_of();
  }
  T err = 0;
  std::vector<int> ir = lrmat.get_ir();
  std::vector<int> ic = lrmat.get_ic();

  for (int j=0;j<lrmat.nb_rows();j++){
    for (int k=0;k<lrmat.nb_cols();k++){
      std::complex<T> aux=ref.get_coefs(ir[j],ic[k])[l];
      for (int r=0;r<reqrank;r++){
        aux = aux - lrmat[l].get_U(j,r) * lrmat[l].get_V(r,k);
      }
      err+=std::pow(std::abs(aux),2);
    }
  }
  return std::sqrt(err);
}

}

#endif
