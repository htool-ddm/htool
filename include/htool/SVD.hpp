#ifndef HTOOL_SVD_HPP
#define HTOOL_SVD_HPP

#include "lrmat.hpp"
#include <cassert>
#include <Eigen/Dense>
// TODO use lapack instead of eigen
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


public:
  SVD(const std::vector<int>& ir0, const std::vector<int>& ic0, int rank0=-1): LowRankMatrix<T>(ir0,ic0,rank0){}

	SVD(const std::vector<int>& ir0, const std::vector<int>& ic0,int offset_i0, int offset_j0, int rank0=-1): LowRankMatrix<T>(ir0,ic0,offset_i0,offset_j0,rank0){}

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
    SubMatrix<T> submat = A.get_submatrix(this->ir,this->ic);
    for (int i=0; i<M.rows(); i++){
  		for (int j=0; j<M.cols(); j++){
  			M(i,j) = submat(i, j);
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

	void build(const IMatrix<T>& A, const Cluster& t, const std::vector<R3> xt,const std::vector<int> tabt, const Cluster& s, const std::vector<R3> xs, const std::vector<int>tabs){
    this->build(A);
  }
  T get_singular_value(int i){return singular_values[i];}

};

}


#endif
