#ifndef HTOOL_PARTIALACA_HPP
#define HTOOL_PARTIALACA_HPP

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
template<typename T>
class partialACA: public LowRankMatrix<T>{


public:
	//=========================//
	//    PARTIAL PIVOT ACA    //
	//=========================//
    // If reqrank=-1 (default value), we use the precision given by epsilon for the stopping criterion;
    // otherwise, we use the required rank for the stopping criterion (!: at the end the rank could be lower)
  	using LowRankMatrix<T>::LowRankMatrix;


	void build(const IMatrix<T>& A, const Cluster& t, const std::vector<R3>& xt,const std::vector<int>& tabt, const Cluster& s, const std::vector<R3>& xs, const std::vector<int>& tabs){
		if(this->rank == 0){
			this->U.resize(this->nr,1);
			this->V.resize(1,this->nc);
		}
		else{

			//// Choice of the first row (see paragraph 3.4.3 page 151 Bebendorf)
			double dist=1e30;
			int I=0;
			for (int i =0;i<int(this->nr/Parametres::ndofperelt);i++){
				double aux_dist= norm2(xt[tabt[this->ir[i*Parametres::ndofperelt]]]-t.get_ctr());
				if (dist>aux_dist){
					dist=aux_dist;
					I=i*Parametres::ndofperelt;
				}
			}
			// Partial pivot
			int J=0;
			int q = 0;
			int reqrank = this->rank;
			std::vector<std::vector<T> > uu, vv;
			std::vector<bool> visited_row(this->nr,false);
			std::vector<bool> visited_col(this->nc,false);

			double frob = 0;
			double aux  = 0;
			// Either we have a required rank
			// Or it is negative and we have to check the relative error between two iterations.
			//But to do that we need a least two iterations.
			while (((reqrank > 0) && (q < std::min(reqrank,std::min(this->nr,this->nc))) ) ||
			       ((reqrank < 0) && (q==0 || sqrt(aux/frob)>this->epsilon))) {

				// Next current rank
				q+=1;

				if (q*(this->nr+this->nc) > (this->nr*this->nc)) { // the next current rank would not be advantageous
                    q=-1;
					break;
				}
				else{
					std::vector<T> r(this->nc),c(this->nr);

					// Compute the first cross
					//==================//
					// Look for a column
					double pivot = 0.;
					SubMatrix<T> row = A.get_submatrix(std::vector<int> {this->ir[I]},this->ic);
					for(int k=0; k<this->nc; k++){
						r[k] = row(0,k);//A.get_coef(this->ir[I],this->ic[k]);
						for(int j=0; j<uu.size(); j++){
							r[k] += -uu[j][I]*vv[j][k];
						}
						if( std::abs(r[k])>pivot && !visited_col[k] ){
							J=k; pivot=std::abs(r[k]);}
					}

					visited_row[I] = true;
					T gamma = T(1.)/r[J];
					//==================//
					// Look for a line
					if( std::abs(r[J]) > 1e-15 ){
						double cmax = 0.;
						SubMatrix<T> col = A.get_submatrix(this->ir,std::vector<int> {this->ic[J]});
						for(int j=0; j<this->nr; j++){
							c[j] = col(j,0);//A.get_coef(this->ir[j],this->ic[J]);
							for(int k=0; k<uu.size(); k++){
								c[j] += -uu[k][j]*vv[k][J];
							}
							c[j] = gamma*c[j];
							if( std::abs(c[j])>cmax && !visited_row[j] ){
								I=j; cmax=std::abs(c[j]);}
						}
						visited_col[J] = true;
						// Test if no given rank
						if (reqrank<0){
							// Error estimator
							T frob_aux = 0.;
							aux = std::abs(dprod(c,c)*dprod(r,r));
							// aux: terme quadratiques du developpement du carre' de la norme de Frobenius de la matrice low rank
							for(int j=0; j<uu.size(); j++){
								frob_aux += dprod(r,vv[j])*dprod(c,uu[j]);
							}
							// frob_aux: termes croises du developpement du carre' de la norme de Frobenius de la matrice low rank
							frob += aux + 2*std::real(frob_aux); // frob: Frobenius norm of the low rank matrix
							//==================//
						}
						// Matrix<T> M=A.get_submatrix(this->ir,this->ic);
						// uu.push_back(M.get_col(J));
						// vv.push_back(M.get_row(I)/M(I,J));
						// New cross added
						uu.push_back(c);
						vv.push_back(r);

					}
					else{
						// std::cout << "There is a zero row in the starting submatrix and ACA didn't work" << std::endl;
						q-=1;
						break;
					}
				}
			}
			// Final rank
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
};
}
#endif
