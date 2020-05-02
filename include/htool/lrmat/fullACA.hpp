#ifndef HTOOL_FULL_ACA_HPP
#define HTOOL_FULL_ACA_HPP

#include <iostream>
#include <fstream>
#include <complex>
#include <vector>
#include <cassert>
#include "lrmat.hpp"


namespace htool {
//================================//
//   CLASSE MATRICE RANG FAIBLE   //
//================================//
//
// Refs biblio:
//
//  -> slides de Stéphanie Chaillat:
//           http://uma.ensta-paristech.fr/var/files/chaillat/seance2.pdf
//           et en particulier la slide 25
//
//  -> livre de M.Bebendorf:
//           http://www.springer.com/kr/book/9783540771463
//           et en particulier le paragraphe 3.4
//
//  -> livre de Rjasanow-Steinbach:
//           http://www.ems-ph.org/books/book.php?proj_nr=125
//           et en particulier le paragraphe 3.2
//
//=================================//
template<typename T, typename ClusterImpl>
class fullACA: public LowRankMatrix<T, ClusterImpl>{

public:
	//=========================//
	//    FULL PIVOT ACA    //
	//=========================//
    // If reqrank=-1 (default value), we use the precision given by epsilon for the stopping criterion;
    // otherwise, we use the required rank for the stopping criterion (!: at the end the rank could be lower)
	using LowRankMatrix<T,ClusterImpl>::LowRankMatrix;

	void build(const IMatrix<T>& A){
		if(this->rank == 0){
			this->U.resize(this->nr,1);
			this->V.resize(1,this->nc);
		}
		else{

			// Matrix assembling
			Matrix<T> M=A.get_submatrix(this->ir,this->ic);

			// Full pivot
			int q=0;
			int reqrank = this->rank;
			std::vector<std::vector<T> > uu;
			std::vector<std::vector<T> > vv;
			double Norm = normFrob(M);

			while (((reqrank > 0) && (q < std::min(reqrank,std::min(this->nr,this->nc))) ) ||
			      ( (reqrank < 0) && ( normFrob(M)/Norm>this->epsilon || q==0) )) {

				q+=1;
				if (q*(this->nr+this->nc) > (this->nr*this->nc)) { // the current rank would not be advantageous
					q=-1;
					break;
				}
				else{
					std::pair<int , int > ind = argmax(M);
					T pivot = M(ind.first,ind.second);
                    if (std::abs(pivot)<1e-15) {
                        q+=-1; break;
                    }
					uu.push_back(M.get_col(ind.second));
					vv.push_back(M.get_row(ind.first)/pivot);

					for (int i =0 ; i<M.nb_rows();i++){
						for (int j =0 ; j<M.nb_cols();j++){
							M(i,j)-=uu[q-1][i]*vv[q-1][j];
						}
					}
				}
			}
			this->rank=q;
			if (this->rank>0){
				this->U.resize(this->nr,this->rank);
				this->V.resize(this->rank,this->nc);
				for (int k=0;k<this->rank;k++){
					this->U.set_col(k,uu[k]);
					this->V.set_row(k,vv[k]);
				}
			}
		}
	}
	void build(const IMatrix<T>& A, const Cluster<ClusterImpl>& t, const std::vector<R3>& xt,const std::vector<int>& tabt, const Cluster<ClusterImpl>& s, const std::vector<R3>& xs, const std::vector<int>& tabs){
    this->build(A);
  }
};

}
#endif
