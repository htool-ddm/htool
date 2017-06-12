#ifndef ACA_HPP
#define ACA_HPP

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


private:
	// No assignement or copy
	partialACA(const partialACA& copy_from);
	partialACA & operator=(const partialACA& copy_from);

public:
	//=========================//
	//    PARTIAL PIVOT ACA    //
	//=========================//
    // If reqrank=-1 (default value), we use the precision given by epsilon for the stopping criterion;
    // otherwise, we use the required rank for the stopping criterion (!: at the end the rank could be lower)
	partialACA(const std::vector<int>& ir0, const std::vector<int>& ic0, const Cluster& t0,const Cluster& s0, int rank0=-1):LowRankMatrix<T>(ir0,ic0,t0,s0,rank0){}



	void build(const IMatrix<T>& A){
		if(this->rank == 0){
			this->U.resize(this->nr,1);
			this->V.resize(1,this->nc);
		}
		else{

			//// Choice of the first row (see paragraph 3.4.3 page 151 Bebendorf)
			double dist=1e30;
			int I=0;
			for (int i =0;i<int(this->nr/this->ndofperelt);i++){
				double aux_dist= norm(this->t.pts_()[this->t.tab_()[this->t.num_()[i*this->ndofperelt]]]-this->t.ctr_());
				if (dist>aux_dist){
					dist=aux_dist;
					I=i*this->ndofperelt;
				}
			}

			// Partial pivot
			int J=0;
			int q = 1;
			int reqrank = this->rank;
			std::vector<std::vector<T> > uu, vv;
			std::vector<bool> visited_row(this->nr,false);
			std::vector<bool> visited_col(this->nc,false);

			double frob = 0.;
			double aux  = 0.;

			while (((reqrank > 0) && (q < reqrank) ) ||
			      ( (reqrank < 0) && ( sqrt(aux/frob)>this->epsilon ) )) {
				if (q*(this->nr+this->nc) > (this->nr*this->nc)) { // the current rank would not be advantageous
					std::cout << "Pas avantageux" << std::endl;
					break;
				}
				else{
					std::vector<T> r(this->nc),c(this->nr);

					// Compute the first cross
					//==================//
					// Recherche colonne
					double pivot = 0.;
					for(int k=0; k<this->nc; k++){
						r[k] = A.get_coef(this->ir[I],this->ic[k]);
						for(int j=0; j<uu.size(); j++){
							r[k] += -uu[j][I]*vv[j][k];
						}
						if( std::abs(r[k])>pivot && !visited_col[k] ){
							J=k; pivot=std::abs(r[k]);}
					}
					visited_row[I] = true;

					//==================//
					// Recherche ligne

					if( std::abs(r[J]) > 1e-15 ){
						double cmax = 0.;
						for(int j=0; j<this->nr; j++){
							c[j] = A.get_coef(this->ir[j],this->ic[J]);
							for(int k=0; k<uu.size(); k++){
								c[j] += -uu[k][j]*vv[k][J];
							}
							if( std::abs(c[j])>cmax && !visited_row[j] ){
								I=j; cmax=std::abs(c[j]);}
						}
						c /= pivot;
						visited_col[J] = true;

						// We accept the cross
						q++;
						//====================//
						// Estimateur d'erreur
						T frob_aux = 0.;
						aux = std::pow(norm2(c)*norm2(r),2);
						// aux: terme quadratiques du developpement du carre' de la norme de Frobenius de la matrice low rank
						for(int j=0; j<uu.size(); j++){
							frob_aux += dprod(r,vv[j])*dprod(c,uu[j]);
						}
						// frob_aux: termes croises du developpement du carre' de la norme de Frobenius de la matrice low rank
						frob += aux + 2*std::real(frob_aux); // frob: Frobenius norm of the low rank matrix
						//==================//
						// Nouvelle croix
						uu.push_back(c);
						vv.push_back(r);
					}
					else{std::cout << "There is a zero row in the starting submatrix and ACA didn't work" << std::endl;
					}
					this->rank=q-1;
					if (this->rank==0){
						this->U.resize(this->nr,1);
						this->V.resize(1,this->nc);
					}
					else{
						this->U.resize(this->nr,this->rank);
						this->V.resize(this->rank,this->nc);
						for (int k=0;k<this->rank;k++){
							this->U.set_col(k,uu[k]);
							this->V.set_row(k,vv[k]);
						}
					}
				}
			}
		}
	}
};
}
#endif
