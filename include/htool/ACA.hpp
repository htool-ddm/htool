#ifndef ACA_HPP
#define ACA_HPP

#include <iostream>
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
class ACA: public LowRankMatrix<T>{


private:
	// No assignement or copy
	ACA(const ACA& copy_from);
	ACA & operator=(const ACA& copy_from);

public:
	//=========================//
	//    PARTIAL PIVOT ACA    //
	//=========================//
    // If reqrank=-1 (default value), we use the precision given by epsilon for the stopping criterion;
    // otherwise, we use the required rank for the stopping criterion (!: at the end the rank could be lower)
	ACA(const std::vector<int>& ir0, const std::vector<int>& ic0, const Cluster& t0,const Cluster& s0, int rank0=-1):LowRankMatrix(ir0,ic0,t0,s0,rank0){}



	void build(const IMatrix<T>& A){
		rank = 0;

		std::vector<vectCplx> u, v;

		std::vector<bool> visited_row(nr,false);
		std::vector<bool> visited_col(nc,false);

		T frob = 0.;
		T aux  = 0.;
		T frob_aux=0;

		//// Choice of the first row (see paragraph 3.4.3 page 151 Bebendorf)
		double dist=1e30;
		int I=0;
		for (int i =0;i<int(nr/ndofperelt);i++){
			double aux_dist= norm(t.pts_()[t.tab_()[t.num_()[i*ndofperelt]]]-t.ctr_());
			if (dist>aux_dist){
				dist=aux_dist;
				I=i*ndofperelt;
			}
		}

		int J=0;
		int q = 0;
		if(reqrank == 0){
			rank = 0; // approximate with a zero matrix
		}
		else if ( (nr+nc)>=(nr*nc) ){ // even rank 1 is not advantageous
			rank=-5; // just a flag for BuildBlockTree (the block won't be treated as a FarField block)
		}
		else{
			std::vector<T> r(nc),c(nr);

			// Compute the first cross
			//==================//
			// Recherche colonne
			T rmax = 0.;
			for(int k=0; k<nc; k++){
				r[k] = A.get_coef(ir[I],ic[k]);
				for(int j=0; j<u.size(); j++){
					r[k] += -u[j][I]*v[j][k];}
				if( abs(r[k])>rmax && !visited_col[k] ){
					J=k; rmax=abs(r[k]);}
			}
			visited_row[I] = true;
			//==================//
			// Recherche ligne
			if( abs(r[J]) > 1e-15 ){
				T gamma = T(1.)/r[J];
				T cmax = 0.;
				for(int j=0; j<nr; j++){
					c[j] = A.get_coef(ir[j],ic[J]);
					for(int k=0; k<u.size(); k++){
						c[j] += -u[k][j]*v[k][J];}
					c[j] = gamma*c[j];
					if( abs(c[j])>cmax && !visited_row[j] ){
						I=j; cmax=abs(c[j]);}
				}
				visited_col[J] = true;

				// We accept the cross
				q++;
				//====================//
				// Estimateur d'erreur
				frob_aux = 0.;
				aux = abs(dprod(c,c)*dprod(r,r));
				// aux: terme quadratiques du developpement du carre' de la norme de Frobenius de la matrice low rank
				for(int j=0; j<u.size(); j++){
					frob_aux += dprod(r,v[j])*dprod(c,u[j]);}
				// frob_aux: termes croises du developpement du carre' de la norme de Frobenius de la matrice low rank
				frob += aux + 2*frob_aux.real(); // frob: Frobenius norm of the low rank matrix
				//==================//
				// Nouvelle croix
				u.push_back(c);
				v.push_back(r);
			}
			else{std::cout << "There is a zero row in the starting submatrix and ACA didn't work" << std::endl;}

			// Stopping criterion of slide 26 of Stephanie Chaillat and Rjasanow-Steinbach
            // (if epsilon>=1, it always stops to rank 1 since frob=aux)
			while ( ((reqrank > 0) && (q < reqrank) ) ||
			       ( (reqrank < 0) && ( sqrt(aux/frob)>epsilon ) ) ) {

				if (q >= std::min(nr,nc) )
					break;
				if ( (q+1)*(nr+nc) > (nr*nc) ){ // one rank more is not advantageous
					if (reqrank <0){ // If we didn't required a rank, i.e. we required a precision with epsilon
						rank=-5;  // a flag for BuildBlockTree to say that the block won't be treated as a FarField block
					}
					break; // If we required a rank, we keep the computed ACA approximation (of lower rank)
				}
				// Compute another cross
				//==================//
				// Recherche colonne
				rmax = 0.;
				for(int k=0; k<nc; k++){
					r[k] = A.get_coef(ir[I],ic[k]);
					for(int j=0; j<u.size(); j++){
						r[k] += -u[j][I]*v[j][k];}
					if( abs(r[k])>rmax && !visited_col[k] ){
						J=k; rmax=abs(r[k]);}
				}
				visited_row[I] = true;
				//==================//
				// Recherche ligne
				if( abs(r[J]) > 1e-15 ){
					T gamma = Cplx(1.)/r[J];
					T cmax = 0.;
					for(int j=0; j<nr; j++){
						c[j] = A.get_coef(ir[j],ic[J]);
						for(int k=0; k<u.size(); k++){
							c[j] += -u[k][j]*v[k][J];}
						c[j] = gamma*c[j];
						if( abs(c[j])>cmax && !visited_row[j] ){
							I=j; cmax=abs(c[j]);}
					}
					visited_col[J] = true;

					aux = abs(dprod(c,c)*dprod(r,r));
                    // aux: terme quadratiques du developpement du carre' de la norme de Frobenius de la matrice low rank
				}
				else{ std::cout << "ACA's loop terminated" << std::endl; break; } // terminate algorithm with exact rank q (not full-rank submatrix)
				// We accept the cross
				q++;
				//====================//
				// Estimateur d'erreur
				frob_aux = 0.;
				for(int j=0; j<u.size(); j++){
					frob_aux += dprod(r,v[j])*dprod(c,u[j]);}
                // frob_aux: termes croises du developpement du carre' de la norme de Frobenius de la matrice low rank
				frob += aux + 2*frob_aux.real(); // frob: Frobenius norm of the low rank matrix
				//==================//
				// Nouvelle croix
				u.push_back(c);
				v.push_back(r);
			}

			U.resize(nr,u.size());
			V.resize(v.size(),nc);
			for (int i=0; i<nr; i++)
			for (int j=0; j<u.size(); j++)
				U(i,j) = u[j][i];

			for (int i=0; i<v.size(); i++)
			for (int j=0; j<nc; j++)
				V(i,j) = v[i][j];

			if (rank != -5)
				rank = u.size();
		}

// Use this for Bebendorf stopping criterion (3.58) pag 141 (not very flexible):
//		if(reqrank == 0)
//			rank = 0; // approximate with a zero matrix
//		else if ( (nr+nc)>=(nr*nc) ){ // even rank 1 is not advantageous
//			rank=-5; // just a flag for BuildBlockTree (the block won't be treated as a FarField block)
//		} else{
//			vectCplx r(nc),c(nr);
//
//			// Compute the first cross
//			// (don't modify the code because we want to really use the Bebendorf stopping criterion (3.58),
//			// i.e. we don't want to accept the new cross if it is not satisfied because otherwise the approximation would be more precise than desired)
//			//==================//
//			// Recherche colonne
//			Real rmax = 0.;
//			for(int k=0; k<nc; k++){
//				r[k] = A.get_coef(ir[I],ic[k]);
//				for(int j=0; j<u.size(); j++){
//					r[k] += -u[j][I]*v[j][k];}
//				if( abs(r[k])>rmax && !visited_col[k] ){
//					J=k; rmax=abs(r[k]);}
//			}
//			visited_row[I] = true;
//			//==================//
//			// Recherche ligne
//			if( abs(r[J]) > 1e-15 ){
//				Cplx gamma = Cplx(1.)/r[J];
//				Real cmax = 0.;
//				for(int j=0; j<nr; j++){
//					c[j] = A.get_coef(ir[j],ic[J]);
//					for(int k=0; k<u.size(); k++){
//						c[j] += -u[k][j]*v[k][J];}
//					c[j] = gamma*c[j];
//					if( abs(c[j])>cmax && !visited_row[j] ){
//						I=j; cmax=abs(c[j]);}
//				}
//				visited_col[J] = true;
//
//				aux = abs(dprod(c,c)*dprod(r,r));
//			}
//			else{std::cout << "There is a zero row in the starting submatrix and ACA didn't work" << std::endl;}
//
//			// (see Bebendorf stopping criterion (3.58) pag 141)
//			while ( (q == 0) ||
//			       ( (reqrank > 0) && (q < reqrank) ) ||
//			       ( (reqrank < 0) && ( sqrt(aux/frob)>Parametres.epsilon * (1 - Parametres.eta)/(1 + Parametres.epsilon) ) ) ) {
//
//				// We accept the cross
//				q++;
//				//====================//
//				// Estimateur d'erreur
//				frob_aux = 0.;
//				//aux = abs(dprod(c,c)*dprod(r,r)); // (already computed to evaluate the test)
//				// aux: terme quadratiques du developpement du carre' de la norme de Frobenius de la matrice low rank
//				for(int j=0; j<u.size(); j++){
//					frob_aux += dprod(r,v[j])*dprod(c,u[j]);}
//				// frob_aux: termes croises du developpement du carre' de la norme de Frobenius de la matrice low rank
//				frob += aux + 2*frob_aux.real(); // frob: Frobenius norm of the low rank matrix
//				//==================//
//				// Nouvelle croix
//				u.push_back(c);
//				v.push_back(r);
//
//				if (q >= std::min(nr,nc) )
//					break;
//				if ( (q+1)*(nr +nc) > (nr*nc) ){ // one rank more is not advantageous
//					if (reqrank <0){ // If we didn't required a rank, i.e. we required a precision with epsilon
//						rank=-5; // a flag for BuildBlockTree to say that the block won't be treated as a FarField block
//					}
//					break; // If we required a rank, we keep the computed ACA approximation (of lower rank)
//				}
//				// Compute another cross
//				//==================//
//				// Recherche colonne
//				rmax = 0.;
//				for(int k=0; k<nc; k++){
//					r[k] = A.get_coef(ir[I],ic[k]);
//					for(int j=0; j<u.size(); j++){
//						r[k] += -u[j][I]*v[j][k];}
//					if( abs(r[k])>rmax && !visited_col[k] ){
//						J=k; rmax=abs(r[k]);}
//				}
//				visited_row[I] = true;
//				//==================//
//				// Recherche ligne
//				if( abs(r[J]) > 1e-15 ){
//					Cplx gamma = Cplx(1.)/r[J];
//					Real cmax = 0.;
//					for(int j=0; j<nr; j++){
//						c[j] = A.get_coef(ir[j],ic[J]);
//						for(int k=0; k<u.size(); k++){
//							c[j] += -u[k][j]*v[k][J];}
//						c[j] = gamma*c[j];
//						if( abs(c[j])>cmax && !visited_row[j] ){
//							I=j; cmax=abs(c[j]);}
//					}
//					visited_col[J] = true;
//
//					aux = abs(dprod(c,c)*dprod(r,r));
//				}
//				else{ std::cout << "ACA's loop terminated" << std::endl; break; } // terminate algorithm with exact rank q (not full-rank submatrix)
//			}
//
//			rank = u.size();
//		}



	}

	ACA(const ACA& m){
		ir=m.ir;
		ic=m.ic;
		nr=m.nr; nc=m.nc; rank = m.rank;
		/*
		u.resize(rank); v.resize(rank);
		for(int j=0; j<rank; j++){
			u[j] = m.u[j]; v[j] = m.v[j];}
		*/
		U.resize(nb_rows(m.U), nb_cols(m.U));
		V.resize(nb_rows(m.V), nb_cols(m.V));
		U = m.U;
		V = m.V;
	}

	void operator=(const ACA& m){
		nr=m.nr; nc=m.nc; rank = m.rank;
		/*
		u.resize(rank); v.resize(rank);
		for(int j=0; j<rank; j++){
			u[j] = m.u[j]; v[j] = m.v[j];}
		*/
		U.resize(nb_rows(m.U), nb_cols(m.U));
		V.resize(nb_rows(m.V), nb_cols(m.V));
		U = m.U;
		V = m.V;
	}

    // 1- !!!
	friend Real CompressionRate(const ACA& m){
		return (1 - ( m.rank*( 1./Real(m.nr) + 1./Real(m.nc)) ));
	}

	//	void Append(const vectCplx& new_u, const vectCplx& new_v){
	//		assert(new_u.size()==nr); u.push_back(new_u);
	//		assert(new_v.size()==nc); v.push_back(new_v);
	//		rank++;
	//	}

	friend Real NormFrob(const ACA& m){
		/*
		const std::vector<vectCplx>& u = m.u;
		const std::vector<vectCplx>& v = m.v;
		const int& rank = m.rank;
		*/

		Cplx frob = 0.;

		for (int j=0;j<m.nr;j++)
		for (int k=0;k<m.nc;k++){
			Cplx aux=0;
				for (int l=0;l<m.rank;l++){
					aux += m.U(j,l) * m.V(l,k);
				}
			frob+=pow(abs(aux),2);
		}

		/*
		for(int j=0; j<rank; j++){
			for(int k=0; k<rank; k++){
				frob += dprod(v[k],v[j])*dprod(u[k],u[j]) ;
			}
		}
		*/
		return sqrt(abs(frob));
	}

	vectCplx operator*(const vectCplx& rhs){
		assert(rhs.size()==nc);
		std::vector<Cplx> tmp(rank);
		tmp = V*rhs;
		vectCplx lhs(nr,0.);
		lhs = U*tmp;
		/*
		for(int k=0; k<v.size(); k++){
			Cplx pk = (v[k],rhs);
			for(int j=0; j<nr; j++){
				lhs[j] += pk*u[k][j];
			}
		}
		*/
		return lhs;
	}

	template <typename LhsType, typename RhsType>
	friend void MvProd(LhsType& lhs, const ACA& m, const RhsType& rhs){
		std::vector<Cplx> tmp(m.rank);
		MvProd(tmp,m.V,rhs);
		MvProd(lhs,m.U,tmp);

		/*
		const std::vector<vectCplx>& u = m.u;
		const std::vector<vectCplx>& v = m.v;
		for(int k=0; k<v.size(); k++){
			Cplx pk = (rhs,v[k]);
			for(int j=0; j<m.nr; j++){
				lhs[j] += pk*u[k][j];
			}
		}
		*/
	}

	friend std::ostream& operator<<(std::ostream& os, const ACA& m){
		os << "rank:\t" << m.rank << std::endl;
		os << "nr:\t"   << m.nr << std::endl;
		os << "nc:\t"   << m.nc << std::endl;
		os << "\nu:\n";
		for(int j=0; j<m.nr; j++){
			for(int k=0; k<m.rank; k++){
				std::cout << m.U(j,k) << "\t";}
			std::cout << "\n";}
		os << "\nv:\n";
		for(int j=0; j<m.nc; j++){
			for(int k=0; k<m.rank; k++){
				std::cout << m.V(k,j) << "\t";
			}
			std::cout << "\n";
		}

		return os;
	}


};


//======================//
//    FULL PIVOT ACA    //
//======================//
/*
 ACA::ACA(const int& rk, const Matrix& A){
 rank = 0;
 nr = nb_rows(A);
 nc = nb_cols(A);
 Matrix R=A;

 Int2 ind = argmax(R);
 int jj=ind.first, kk=ind.second;
 Real rmax=0.;

 for(int q=0; q<rk; q++){

 if( abs(R(jj,kk))<1e-10 ){
 break;}
 else{
 Cplx gamma = 1./R(jj,kk);
 vectCplx c =  gamma*col(A,kk);
 vectCplx r =  row(A,jj);

 rmax = 0.;
 for(int j=0; j<nr; j++){
	for(int k=0; k<nc; k++){
 R(j,k) = R(j,k) - c[j]*r[k];
 if(abs(R(j,k))>rmax){
 rmax=abs(R(j,k)); jj=j; kk=k;}
	}
 }

 u.push_back(c);
 v.push_back(r);
 rank++;
 }

 }

 }
 ==========================*/
}
#endif
