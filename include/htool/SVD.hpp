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
  SVD(const std::vector<int>& ir0, const std::vector<int>& ic0, const Cluster& t0, const Cluster& s0, int rank0=0): LowRankMatrix<T>(ir0,ic0,t0,s0,rank0){}

  void build(const IMatrix<T>& A){
    //// Matrix assembling
    DenseMatrix M(this->ir.size(),this->ic.size());
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

    this->U.resize(M.rows(),sv.size());
    this->V.resize(sv.size(),M.cols());

    for (int i=0;i<M.rows();i++){
      for (int j=0;j<sv.size();j++){
        this->U(i,j)=uu(i,j)*sv[j];
      }
    }
    for (int i=0;i<sv.size();i++){
      for (int j=0;j<M.cols();j++){
          this->V(i,j)=vv(j,i);
        }
    }
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
