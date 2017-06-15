#ifndef SVD_HPP
#define SVD_HPP

#include "lrmat.hpp"
#include <cassert>
#include <Eigen/Dense>

namespace htool{

template< typename T >
class SVD: public LowRankMatrix<T>{

private:
  // Eigen
  static const int Dynamic = Eigen::Dynamic;
  typedef Eigen::Matrix<T, Dynamic, Dynamic>     DenseMatrix;
  typedef Eigen::JacobiSVD<DenseMatrix>          SVDType;
  typedef typename SVDType::SingularValuesType   SgValType;
  typedef typename SVDType::MatrixUType		       UMatrixType;
  typedef typename SVDType::MatrixVType		       VMatrixType;

  // Data member
  std::vector<T> singular_values;

  // No assignement or copy
  SVD(const SVD& copy_from);
  SVD & operator=(const SVD& copy_from);

public:
  SVD(const std::vector<int>& ir0, const std::vector<int>& ic0, int rank0=-1): LowRankMatrix<T>(ir0,ic0,rank0){}

  void build(const IMatrix<T>& A){
    int reqrank=0;
    if (this->rank==0){
      this->U.resize(this->nr,1);
      this->V.resize(1,this->nc);
      return;
    }
    else if (this->rank==-1){
      reqrank=std::min(this->ir.size(),this->ic.size());
      this->rank=reqrank;
    }
    else{
      reqrank=this->rank;
    }
    //// Matrix assembling
    DenseMatrix M(this->nr,this->nc);
    for (int i=0; i<M.rows(); i++){
  		for (int j=0; j<M.cols(); j++){
  			M(i,j) = A.get_coef(this->ir[i], this->ic[j]);
      }
    }

    //// SVD
    SVDType svd(M,Eigen::ComputeThinU | Eigen::ComputeThinV );

    const SgValType& sv = svd.singularValues();
    singular_values.resize(sv.size());
    for(int j=0; j<sv.size(); j++){singular_values[j]=sv[j];}
    const UMatrixType& uu = svd.matrixU();
    const VMatrixType& vv = svd.matrixV();

    this->U.resize(M.rows(),reqrank);
    this->V.resize(reqrank,M.cols());

    for (int i=0;i<M.rows();i++){
      for (int j=0;j<reqrank;j++){
        this->U(i,j)=uu(i,j)*sv[j];
      }
    }
    for (int i=0;i<reqrank;i++){
      for (int j=0;j<M.cols();j++){
          this->V(i,j)=vv(j,i);
        }
    }
  }

  void build(const IMatrix<T>& A, const Cluster& t, const Cluster& s){
    this->build(A);
  }
  T get_singular_value(int i){return singular_values[i];}
  //
  // double NormFrob(){
  //   T frob = 0.;
  //
  //   for (int j=0;j<m.nr;j++){
  //     for (int k=0;k<m.nc;k++){
  //       T aux=0;
  //       for (int l=0;l<m.rank;l++){
  //         T += m.U(j,l) * m.V(l,k);
  //       }
  //     frob+=pow(abs(aux),2);
  //     }
  //   }
  // }
};

}


#endif
