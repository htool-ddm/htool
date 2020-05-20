#ifndef HTOOL_LRMAT_HPP
#define HTOOL_LRMAT_HPP

#include <vector>
#include "../clustering/cluster.hpp"
#include <htool/clustering/ncluster.hpp>
#include "../types/matrix.hpp"
#include "../types/multimatrix.hpp"
namespace htool{

template<typename T, typename ClusterImpl=GeometricClustering>
class LowRankMatrix: public Parametres{

protected:
    // Data member
    int rank, nr, nc;
    Matrix<T>  U,V;
    std::vector<int> ir;
    std::vector<int> ic;
    int offset_i;
    int offset_j;
public:

    // Constructors
    LowRankMatrix() = delete;
    LowRankMatrix(const std::vector<int>& ir0, const std::vector<int>& ic0, int rank0=-1):rank(rank0), nr(ir0.size()), nc(ic0.size()), U(ir0.size(),1),V(1,ic0.size()), ir(ir0), ic(ic0), offset_i(0), offset_j(0){}
    LowRankMatrix(const std::vector<int>& ir0, const std::vector<int>& ic0, int offset_i0, int offset_j0, int rank0=-1):rank(rank0), nr(ir0.size()), nc(ic0.size()), U(ir0.size(),1),V(1,ic0.size()), ir(ir0),ic(ic0),offset_i(offset_i0), offset_j(offset_j0){}

    // VIrtual function
    virtual void build(const IMatrix<T>& A, const Cluster<ClusterImpl>& t, const std::vector<R3>& xt,const std::vector<int>& tabt, const Cluster<ClusterImpl>& s, const std::vector<R3>& xs, const std::vector<int>& tabs) = 0;

    // Getters
    int nb_rows() const {return this->nr;}
    int nb_cols() const{return this->nc;}
    int rank_of() const {return this->rank;}
    std::vector<int> get_ir() const {return this->ir;}
    std::vector<int> get_ic() const {return this->ic;}
    int get_offset_i() const {return this->offset_i;}
    int get_offset_j() const {return this->offset_j;}
    T get_U(int i, int j) const {return this->U(i,j);}
    T get_V(int i, int j) const {return this->V(i,j);}
    std::vector<int> get_xr() const {return this->xr;}
    std::vector<int> get_xc() const {return this->xc;}
    std::vector<int> get_tabr() const {return this->tabr;}
    std::vector<int> get_tabc() const {return this->tabc;}

    std::vector<T> operator*(const std::vector<T>& a) const{
        return this->U*(this->V*a);
    }
    void mvprod(const T* const in,  T* const out) const{
        if (rank==0){
          std::fill(out,out+nr,0);
        }
        else{
          std::vector<T> a(this->rank);
          V.mvprod(in,a.data());
          U.mvprod(a.data(),out);
        }
    }

    void add_mvprod_row_major(const T* const in,  T* const out, const int& mu, char trans = 'N') const{
        if (rank!=0){
            std::vector<T> a(this->rank*mu);
            if (trans == 'N'){
                V.mvprod_row_major(in,a.data(),mu);
                U.add_mvprod_row_major(a.data(),out,mu);
            }
            else if (trans == 'C'){
                U.mvprod_row_major(in,a.data(),mu,trans);
                V.add_mvprod_row_major(a.data(),out,mu,trans);
            }


        }
    }

    void get_whole_matrix(T* const out) const {
        char transa ='N';
        char transb ='N';
        int M = U.nb_rows();
        int N = V.nb_cols();
        int K = U.nb_cols();
        T alpha = 1;
        int lda =  U.nb_rows();
        int ldb =  V.nb_rows();
        T beta = 0;
        int ldc = U.nb_rows();


        Blas<T>::gemm(&transa, &transb, &M, &N, &K, &alpha, &(U(0,0)),
        &lda, &(V(0,0)), &ldb, &beta, out,&ldc);
    }

    double compression() const{
        return (1 - ( this->rank*( 1./double(this->nr) + 1./double(this->nc))));
    }


    friend std::ostream& operator<<(std::ostream& os, const LowRankMatrix& m){
        os << "rank:\t" << m.rank << std::endl;
        os << "nr:\t"   << m.nr << std::endl;
        os << "nc:\t"   << m.nc << std::endl;
        os << "U:\n";
        os<< m.U << std::endl;
        os<< m.V << std::endl;

        return os;
    }
};

template<typename T, typename ClusterImpl>
double Frobenius_relative_error(const LowRankMatrix<T,ClusterImpl>& lrmat, const IMatrix<T>& ref, int reqrank=-1){
  assert(reqrank<=lrmat.rank_of());
  if (reqrank==-1){
    reqrank=lrmat.rank_of();
  }
  T norm= 0;
  T err = 0;
  std::vector<int> ir = lrmat.get_ir();
  std::vector<int> ic = lrmat.get_ic();

  for (int j=0;j<lrmat.nb_rows();j++){
    for (int k=0;k<lrmat.nb_cols();k++){
      T aux=ref.get_coef(ir[j],ic[k]);
      norm+= std::pow(std::abs(aux),2);
      for (int l=0;l<reqrank;l++){
        aux = aux - lrmat.get_U(j,l) * lrmat.get_V(l,k);
      }
      err+=std::pow(std::abs(aux),2);
    }
  }
  err =err/norm;
  return std::sqrt(err);
}


template<typename T, typename ClusterImpl>
double Frobenius_absolute_error(const LowRankMatrix<T, ClusterImpl>& lrmat, const IMatrix<T>& ref, int reqrank=-1){
  assert(reqrank<=lrmat.rank_of());
  if (reqrank==-1){
    reqrank=lrmat.rank_of();
  }
  T err = 0;
  std::vector<int> ir = lrmat.get_ir();
  std::vector<int> ic = lrmat.get_ic();

  for (int j=0;j<lrmat.nb_rows();j++){
    for (int k=0;k<lrmat.nb_cols();k++){
      T aux=ref.get_coef(ir[j],ic[k]);
      for (int l=0;l<reqrank;l++){
        aux = aux - lrmat.get_U(j,l) * lrmat.get_V(l,k);
      }
      err+=std::pow(std::abs(aux),2);
    }
  }
  return std::sqrt(err);
}

template<typename T, typename ClusterImpl>
double Frobenius_absolute_error(const LowRankMatrix<std::complex<T>,ClusterImpl>& lrmat, const IMatrix<std::complex<T>>& ref, int reqrank=-1){
  assert(reqrank<=lrmat.rank_of());
  if (reqrank==-1){
    reqrank=lrmat.rank_of();
  }
  T err = 0;
  std::vector<int> ir = lrmat.get_ir();
  std::vector<int> ic = lrmat.get_ic();

  for (int j=0;j<lrmat.nb_rows();j++){
    for (int k=0;k<lrmat.nb_cols();k++){
      std::complex<T> aux=ref.get_coef(ir[j],ic[k]);
      for (int l=0;l<reqrank;l++){
        aux = aux - lrmat.get_U(j,l) * lrmat.get_V(l,k);
      }
      err+=std::pow(std::abs(aux),2);
    }
  }
  return std::sqrt(err);
}

template<typename T, typename ClusterImpl>
double Frobenius_absolute_error(const LowRankMatrix<T,ClusterImpl>& lrmat, const MultiIMatrix<T>& ref, int l, int reqrank=-1){
  assert(reqrank<=lrmat.rank_of());
  if (reqrank==-1){
    reqrank=lrmat.rank_of();
  }
  T err = 0;
  std::vector<int> ir = lrmat.get_ir();
  std::vector<int> ic = lrmat.get_ic();

  for (int j=0;j<lrmat.nb_rows();j++){
    for (int k=0;k<lrmat.nb_cols();k++){
      T aux=ref.get_coefs(ir[j],ic[k])[l];
      for (int l=0;l<reqrank;l++){
        aux = aux - lrmat.get_U(j,l) * lrmat.get_V(l,k);
      }
      err+=std::pow(std::abs(aux),2);
    }
  }
  return std::sqrt(err);
}

template<typename T, typename ClusterImpl>
double Frobenius_absolute_error(const LowRankMatrix<std::complex<T>,ClusterImpl>& lrmat, const MultiIMatrix<std::complex<T>>& ref, int l,int reqrank=-1){
  assert(reqrank<=lrmat.rank_of());
  if (reqrank==-1){
    reqrank=lrmat.rank_of();
  }
  T err = 0;
  std::vector<int> ir = lrmat.get_ir();
  std::vector<int> ic = lrmat.get_ic();

  for (int j=0;j<lrmat.nb_rows();j++){
    for (int k=0;k<lrmat.nb_cols();k++){
      std::complex<T> aux=ref.get_coefs(ir[j],ic[k])[l];
      for (int l=0;l<reqrank;l++){
        aux = aux - lrmat.get_U(j,l) * lrmat.get_V(l,k);
      }
      err+=std::pow(std::abs(aux),2);
    }
  }
  return std::sqrt(err);
}



}

#endif
