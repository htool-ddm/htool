#ifndef HTOOL_SVD_HPP
#define HTOOL_SVD_HPP

#include "lrmat.hpp"
#include <cassert>
#include "../wrappers/wrapper_lapack.hpp"

namespace htool{

template< typename T, typename ClusterImpl>
class SVD: public LowRankMatrix<T,ClusterImpl>{

private:
  // Data member
  std::vector<underlying_type<T>> singular_values;


public:
  using LowRankMatrix<T,ClusterImpl>::LowRankMatrix;

  void build(const IMatrix<T>& A){
    int reqrank=0;
    if (this->rank==0){
      this->U.resize(this->nr,1);
      this->V.resize(1,this->nc);
      return;
    }
    else {
      //// Matrix assembling
      double Norm = 0;
      SubMatrix<T> submat = A.get_submatrix(this->ir,this->ic);
      for (int i=0; i<submat.nb_rows(); i++){
        for (int j=0; j<submat.nb_cols(); j++){
          Norm+= std::pow(std::abs(submat(i,j)),2);
        }
      }
      Norm=sqrt(Norm);

      //// SVD
      int m = submat.nb_rows();
      int n = submat.nb_cols();
      int lda = m;
      int ldu = m;
      int ldvt = n;
      int lwork =-1;
      int info;
      singular_values.resize(std::min(m,n));
      Matrix<T> u(m,m);
      // std::vector<T> vt (n*n);
      Matrix<T> vt(n,n);
      std::vector<T> work(std::min(m,n));
      std::vector<underlying_type<T>> rwork(5*std::min(m,n));

      Lapack<T>::gesvd("A","A",&m,&n,submat.data(),&lda,singular_values.data(),u.data(),&ldu,vt.data(),&ldvt,work.data(),&lwork,rwork.data(),&info);
      lwork = (int)std::real(work[0]);
      work.resize(lwork);
      Lapack<T>::gesvd("A","A",&m,&n,submat.data(),&lda,singular_values.data(),u.data(),&ldu,vt.data(),&ldvt,work.data(),&lwork,rwork.data(),&info);

      if (this->rank==-1){

        // Compute Frobenius norm of the approximation error
        int j=singular_values.size();
        double svd_norm=0;

        do{
          j=j-1;
          svd_norm+=std::pow(std::abs(singular_values[j]),2);
        } while (j>0 && std::sqrt(svd_norm)/Norm<this->epsilon);
        
        
        reqrank=std::min(j+1,std::min(m,n));
        
        if (reqrank*(this->nr+this->nc) > (this->nr*this->nc)){
          reqrank=-1;
        }
        this->rank=reqrank;

      }
      else{
        reqrank=std::min(this->rank,std::min(this->nr,this->nc));
      }

      if (this->rank>0){
        this->U.resize(submat.nb_rows(),reqrank);
        this->V.resize(reqrank,submat.nb_cols());

        for (int i=0;i<submat.nb_rows();i++){
          for (int j=0;j<reqrank;j++){
            this->U(i,j)=u(i,j)*singular_values[j];
          }
        }
        for (int i=0;i<reqrank;i++){
          for (int j=0;j<submat.nb_cols();j++){
              this->V(i,j)=vt(i,j);
            }
        }
      }
    }
  }

	void build(const IMatrix<T>& A, const Cluster<ClusterImpl>& t, const std::vector<R3>& xt,const std::vector<int>& tabt, const Cluster<ClusterImpl>& s, const std::vector<R3>& xs, const std::vector<int>& tabs){
    this->build(A);
  }
  T get_singular_value(int i){return singular_values[i];}

};

}


#endif
