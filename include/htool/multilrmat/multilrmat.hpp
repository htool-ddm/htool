#ifndef HTOOL_MULTI_LRMAT_HPP
#define HTOOL_MULTI_LRMAT_HPP

#include <vector>
#include "../clustering/cluster.hpp"
#include "../types/multimatrix.hpp"
#include "../lrmat/barelrmat.hpp"
namespace htool{

template<typename T, typename ClusterImpl>
class MultiLowRankMatrix: public Parametres{

protected:
    // Data member
    int rank, nr, nc, nm;
    // Matrix<T>  U,V;
    std::vector<bareLowRankMatrix<T,ClusterImpl>> LowRankMatrices;
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
    virtual void build(const MultiIMatrix<T>& A, const Cluster<ClusterImpl>& t, const std::vector<R3>& xt,const std::vector<int>& tabt, const Cluster<ClusterImpl>& s, const std::vector<R3>& xs, const std::vector<int>& tabs) = 0;

    // Getters
    int nb_rows()   const {return this->nr;}
    int nb_cols()   const {return this->nc;}
    int nb_lrmats() const {return this->nm;}
    int rank_of() const {return this->rank;}
    std::vector<int> get_ir() const {return this->ir;}
    std::vector<int> get_ic() const {return this->ic;}
    int get_offset_i() const {return this->offset_i;}
    int get_offset_j() const {return this->offset_j;}

    bareLowRankMatrix<T,ClusterImpl>&  operator[](int j){return LowRankMatrices[j];}; 
    const bareLowRankMatrix<T,ClusterImpl>&  operator[](int j) const {return LowRankMatrices[j];}; 


};

template<typename T, typename ClusterImpl>
std::vector<double> Frobenius_absolute_error(const MultiLowRankMatrix<T,ClusterImpl>& lrmat, const MultiIMatrix<T>& ref, int reqrank=-1){
  assert(reqrank<=lrmat[0].rank_of());
  if (reqrank==-1){
      reqrank=lrmat[0].rank_of();
  }
  std::vector<T> err (lrmat.nb_lrmats(),0);
  std::vector<int> ir = lrmat.get_ir();
  std::vector<int> ic = lrmat.get_ic();
  std::vector<T> aux(lrmat.nb_lrmats());

  for (int j=0;j<lrmat.nb_rows();j++){
    for (int k=0;k<lrmat.nb_cols();k++){
      aux=ref.get_coefs(ir[j],ic[k]);
      for (int l=0;l<lrmat.nb_lrmats();l++){
        for (int r=0;r<reqrank;r++){
          aux[l] = aux[l] - lrmat[l].get_U(j,r) * lrmat[l].get_V(r,k);
        }
        err[l]+=std::pow(std::abs(aux[l]),2);
      }
    }
  }

  std::transform(err.begin(), err.end(), err.begin(), (double(*)(double)) sqrt);
  return err;
}

template<typename T, typename ClusterImpl>
std::vector<double> Frobenius_absolute_error(const MultiLowRankMatrix<std::complex<T>,ClusterImpl>& lrmat, const MultiIMatrix<std::complex<T>>& ref, int reqrank=-1){
  assert(reqrank<=lrmat[0].rank_of());

  std::vector<T> err (lrmat.nb_lrmats(),0);
  std::vector<int> ir = lrmat.get_ir();
  std::vector<int> ic = lrmat.get_ic();
  std::vector<std::complex<T>> aux(lrmat.nb_lrmats());
double test_time = MPI_Wtime();
  for (int j=0;j<lrmat.nb_rows();j++){
    for (int k=0;k<lrmat.nb_cols();k++){
      aux=ref.get_coefs(ir[j],ic[k]);
      for (int l=0;l<lrmat.nb_lrmats();l++){
        if (reqrank==-1){
          reqrank=lrmat.rank_of();
        }
        for (int r=0;r<reqrank;r++){
          aux[l] = aux[l] - lrmat[l].get_U(j,r) * lrmat[l].get_V(r,k);
        }
        err[l]+=std::pow(std::abs(aux[l]),2);
      }
    }
  }

  std::transform(err.begin(), err.end(), err.begin(), (double(*)(double)) sqrt);
  return err;
}

}

#endif
