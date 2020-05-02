#ifndef HTOOL_SYMPARTIALACA_HPP
#define HTOOL_SYMPARTIALACA_HPP

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
class sympartialACA: public LowRankMatrix<T,ClusterImpl>{


public:
	//=========================//
	//    PARTIAL PIVOT ACA    //
	//=========================//
    // If reqrank=-1 (default value), we use the precision given by epsilon for the stopping criterion;
    // otherwise, we use the required rank for the stopping criterion (!: at the end the rank could be lower)
  	using LowRankMatrix<T,ClusterImpl>::LowRankMatrix;


	void build(const IMatrix<T>& A, const Cluster<ClusterImpl>& t, const std::vector<R3>& xt,const std::vector<int>& tabt, const Cluster<ClusterImpl>& s, const std::vector<R3>& xs, const std::vector<int>& tabs){
		if(this->rank == 0){
			this->U.resize(this->nr,1);
			this->V.resize(1,this->nc);
		}
		else{
			
			int n1,n2;
			std::vector<int> const * i1;
			std::vector<int> const * i2;
			std::vector<int> const * tab1;
			std::vector<int> const * tab2;
			std::vector<R3> const* x1;
			std::vector<R3> const* x2;
			Cluster<ClusterImpl> const * cluster_1; 
			Cluster<ClusterImpl> const * cluster_2;


			if (this->offset_i>=this->offset_j){

				n1=this->nr;
				n2=this->nc;
				i1=&(this->ir);
				i2=&(this->ic);
				tab1=&tabt;
				tab2=&tabs;
				x1=&xt;
				x2=&xs;
				cluster_1=&t;
				cluster_2=&s;
			}
			else{
				n1=this->nc;
				n2=this->nr;
				i1=&(this->ic);
				i2=&(this->ir);
				tab1=&tabs;
				tab2=&tabt;
				x1=&xs;
				x2=&xt;
				cluster_1=&s;
				cluster_2=&t;
			}


			//// Choice of the first row (see paragraph 3.4.3 page 151 Bebendorf)
			double dist=1e30;
			int I1=0;
			for (int i =0;i<int(n1/Parametres::ndofperelt);i++){
				double aux_dist= norm2((*x1)[(*tab1)[(*i1)[i*Parametres::ndofperelt]]]-(*cluster_1).get_ctr());
				if (dist>aux_dist){
					dist=aux_dist;
					I1=i*Parametres::ndofperelt;
				}
			}
			// Partial pivot
			int I2=0;
			int q = 0;
			int reqrank = this->rank;
			std::vector<std::vector<T> > uu, vv;
			std::vector<bool> visited_1(n1,false);
			std::vector<bool> visited_2(n2,false);

			double frob = 0;
			double aux  = 0;

			// Either we have a required rank
			// Or it is negative and we have to check the relative error between two iterations.
			//But to do that we need a least two iterations.
			while (((reqrank > 0) && (q < std::min(reqrank,std::min(this->nr,this->nc))) ) ||
			       ((reqrank < 0) && (sqrt(aux/frob)>this->epsilon || q==0))) {

				// Next current rank
				q+=1;

				if (q*(this->nr+this->nc) > (this->nr*this->nc)) { // the next current rank would not be advantageous
                    q=-1;
					break;
				}
				else{
					std::vector<T> line2(n2),line1(n1);

					// Compute the first cross
					//==================//
					// Look for a column
					double pivot = 0.;
					if (this->offset_i>=this->offset_j){
						SubMatrix<T> submat1 = A.get_submatrix(std::vector<int> {(*i1)[I1]},*i2);
						for(int k=0; k<n2; k++){
							line2[k] = submat1(0,k);//A.get_coef(this->ir[I],this->ic[k]);
							for(int j=0; j<uu.size(); j++){
								line2[k] += -uu[j][I1]*vv[j][k];
							}
							if( std::abs(line2[k])>pivot && !visited_2[k] ){
								I2=k; pivot=std::abs(line2[k]);}
						}
					}
					else{
						SubMatrix<T> submat1 = A.get_submatrix(*i2,std::vector<int> {(*i1)[I1]});
						for(int k=0; k<n2; k++){
							line2[k] = submat1(k,0);//A.get_coef(this->ir[I],this->ic[k]);
							for(int j=0; j<uu.size(); j++){
								line2[k] += -uu[j][I1]*vv[j][k];
							}
							if( std::abs(line2[k])>pivot && !visited_2[k] ){
								I2=k; pivot=std::abs(line2[k]);}
						}
					}
					visited_1[I1] = true;
					T gamma = T(1.)/line2[I2];
					
					//==================//
					// Look for a line
					if( std::abs(line2[I2]) > 1e-15 ){
						double cmax = 0.;
						if (this->offset_i>=this->offset_j){
							SubMatrix<T> submat2 = A.get_submatrix(*i1,std::vector<int> {(*i2)[I2]});
							for(int j=0; j<n1; j++){
								line1[j] = submat2(j,0);//A.get_coef(this->ir[j],this->ic[J]);
								for(int k=0; k<uu.size(); k++){
									line1[j] += -uu[k][j]*vv[k][I2];
								}
								line1[j] = gamma*line1[j];
								if( std::abs(line1[j])>cmax && !visited_1[j] ){
									I1=j; cmax=std::abs(line1[j]);}
							}
						}
						else{
							SubMatrix<T> submat2 = A.get_submatrix(std::vector<int> {(*i2)[I2]},*i1);
							for(int j=0; j<n1; j++){
								line1[j] = submat2(0,j);//A.get_coef(this->ir[j],this->ic[J]);
								for(int k=0; k<uu.size(); k++){
									line1[j] += -uu[k][j]*vv[k][I2];
								}
								line1[j] = gamma*line1[j];
								if( std::abs(line1[j])>cmax && !visited_1[j] ){
									I1=j; cmax=std::abs(line1[j]);}
							}
						}
						visited_2[I2] = true;

						// Test if no given rank
						if (reqrank<0){
							// Error estimator
							T frob_aux = 0.;
							aux = std::abs(dprod(line2,line2)*dprod(line1,line1));
							// aux: terme quadratiques du developpement du carre' de la norme de Frobenius de la matrice low rank
							for(int j=0; j<uu.size(); j++){
								frob_aux += dprod(line2,vv[j])*dprod(line1,uu[j]);
							}
							// frob_aux: termes croises du developpement du carre' de la norme de Frobenius de la matrice low rank
							frob += aux + 2*std::real(frob_aux); // frob: Frobenius norm of the low rank matrix
							//==================//
						}
						// Matrix<T> M=A.get_submatrix(this->ir,this->ic);
						// uu.push_back(M.get_col(J));
						// vv.push_back(M.get_row(I)/M(I,J));
						// New cross added
						uu.push_back(line1);
						vv.push_back(line2);

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
					if (this->offset_i>=this->offset_j){
						this->U.set_col(k,uu[k]);
						this->V.set_row(k,vv[k]);
					}
					else{
						this->U.set_col(k,vv[k]);
						this->V.set_row(k,uu[k]);
					}
				}
			}
		}
	}
};
}
#endif
