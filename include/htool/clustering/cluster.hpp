#ifndef HTOOL_CLUSTER_HPP
#define HTOOL_CLUSTER_HPP

#include "../types/matrix.hpp"
#include "../types/point.hpp"
#include "../misc/parametres.hpp"
#include <iomanip>
// #include <Eigen/Dense>
// #include <Eigen/Eigenvalues>

namespace htool {
//===============================//
//           PAQUETS             //
//===============================//
//
// Refs biblio:
//
//  -> livre de Sauter-Schwab:
//           http://www.springer.com/kr/book/9783540680925
//           et en particulier le paragraphe 7.1.2
//
//  -> livre de Borm:
//           http://www.ems-ph.org/books/book.php?proj_nr=125
//           et en particulier less paragraphes 3.1, 3.2 et 3.3
//
//  -> livre de Rjasanow-Steinbach:
//           http://www.ems-ph.org/books/book.php?proj_nr=125
//           et en particulier le paragraphe 3.1
//
//=================================//



class Cluster: public Parametres{
private:
	Cluster*        son[2];  // Paquets enfants
	R3              ctr;     // Centre du paquet
	double            rad;     // Rayon du champ proche

	int				rank;    // rang du processeur qui s'occupe des dofs de ce cluster

	int depth; // profondeur du cluster dans l'arbre des paquets

	int max_depth;
	int min_depth;

	int offset;
	int size;

	Cluster & operator=(const Cluster& copy_from); // pas d'affectation


public:
	// build cluster tree
	void build(const std::vector<R3>& x0, const std::vector<double>& r0,const std::vector<int>& tab0, const std::vector<double>& g0, std::vector<int>& perm);

	// Root constructor
	Cluster(const std::vector<R3>& x0, const std::vector<double>& r0,const std::vector<int>& tab0, const std::vector<double>& g0, std::vector<int>& perm);

	Cluster(const std::vector<R3>& x0, const std::vector<int>& tab0, const std::vector<double>& g0, std::vector<int>& perm);
	Cluster(const std::vector<R3>& x0, const std::vector<double>& r0, const std::vector<int>& tab0,std::vector<int>& perm);
	Cluster(const std::vector<R3>& x0, const std::vector<double>& r0, const std::vector<double>& g0, std::vector<int>& perm);

  Cluster(const std::vector<R3>& x0,const std::vector<int>& tab0, std::vector<int>& perm);

	Cluster(const std::vector<R3>& x0, std::vector<int>& perm);

	// Node constructor
	Cluster(const int& dep):ctr(), rad(0.),max_depth(-1),min_depth(-1), offset(0) {
		son[0]=0;son[1]=0; depth = dep;
	}

    ~Cluster(){if (son[0]!=NULL){ delete son[0];son[0]=nullptr;}if (son[1]!=NULL){ delete son[1];son[1]=nullptr;}};


	bool IsLeaf() const { if(son[0]==NULL){return true;} return false; }

	//// Getters
	const double&           get_rad() const {return rad;}
	const R3&               get_ctr() const {return ctr;}
	const Cluster&       		get_son(const int& j) const {return *(son[j]);}
	Cluster&       					get_son(const int& j){return *(son[j]);}
	int     								get_depth() const {return depth;}
	int 										get_rank()const {return rank;}
	int											get_max_depth() const {return max_depth;}
	int											get_min_depth() const {return min_depth;}
	int                     get_offset() const {return offset;}
	int                     get_size() const {return size;}

	//// Setters
	void set_rank  (const int& rank0)   {rank = rank0;}
	void set_offset(const int& offset0) {offset=offset0;}
	void set_size(const int& size0) {size=size0;}

	void print(const std::vector<int>& perm) const;
};


// Build cluster tree
void Cluster::build(const std::vector<R3>& x, const std::vector<double>& r, const std::vector<int>& tab, const std::vector<double>& g, std::vector<int>& perm){
	assert(tab.size()==x.size()*ndofperelt);
	assert(x.size()==g.size());
	assert(x.size()==r.size());
	assert(tab.size()==perm.size());

	// Initialisation
	rad = 0;
	depth = 0;
	max_depth = 0;
	min_depth = -1;
	offset=0;
	size = tab.size();
	son[0]=NULL;son[1]=NULL;
	depth = 0; // ce constructeur est appele' juste pour la racine
	perm.resize(tab.size());
	std::iota(perm.begin(),perm.end(),0); // perm[i]=i


	// Recursion
	std::stack<Cluster*> s;
	std::stack<std::vector<int>> n;
	s.push(this);
	n.push(perm);

	while(!s.empty()){
		Cluster* curr = s.top();
		std::vector<int> num = n.top();
		s.pop();
		n.pop();

		// Mass of the cluster
		int nb_pt = curr->size;
		double G=0;
		for(int j=0; j<nb_pt; j++){
			G += g[tab[num[j]]];
		}

		// Center of the cluster
		R3 xc;
		xc.fill(0);
		for(int j=0; j<nb_pt; j++){
			xc += g[tab[num[j]]]*x[tab[num[j]]];
		}
		xc = (1./G)*xc;
		curr->ctr = xc;

		// Calcul matrice de covariance
		Matrix<double> cov(3,3);
		curr->rad=0.;
		for(int j=0; j<nb_pt; j++){
			R3 u = x[tab[num[j]]] - xc;
			curr->rad=std::max(curr->rad,norm2(u)+r[tab[num[j]]]);
			for(int p=0; p<3; p++){
				for(int q=0; q<3; q++){
					cov(p,q) += g[tab[num[j]]]*u[p]*u[q];
				}
			}
		}

		// Calcul direction principale
		double p1 = pow(cov(0,1),2) + pow(cov(0,2),2) + pow(cov(1,2),2);
		std::vector<double> eigs(3);
		Matrix<double> I(3,3);I(0,0)=1;I(1,1)=1;I(2,2)=1;
		R3 dir;
		if (p1 < 1e-16) {
	    	// cov is diagonal.
	   		eigs[0] = cov(0,0);
	   		eigs[1] = cov(1,1);
	   		eigs[2] = cov(2,2);
			dir[0]=1;dir[1]=0;dir[2]=0;
	   		if (eigs[0] < eigs[1]) {
	   			double tmp = eigs[0];
	   			eigs[0] = eigs[1];
	   			eigs[1] = tmp;
				dir[0]=0;dir[1]=1;dir[2]=0;
	   		}
	   		if (eigs[0] < eigs[2]) {
	   			double tmp = eigs[0];
	   			eigs[0] = eigs[2];
	   			eigs[2] = tmp;
				dir[0]=0;dir[1]=0;dir[2]=1;
	 		}
		}
		else {
			double q = (cov(0,0)+cov(1,1)+cov(2,2))/3.;
	   	double p2 = pow(cov(0,0) - q,2) + pow(cov(1,1) - q,2) + pow(cov(2,2) - q,2) + 2. * p1;
	   	double p = sqrt(p2 / 6.);
	   	Matrix<double> B(3,3);
			B = (1. / p) * (cov - q * I);
	   	double detB = B(0,0)*(B(1,1)*B(2,2)-B(1,2)*B(2,1))
	   					- B(0,1)*(B(1,0)*B(2,2)-B(1,2)*B(2,0))
	   					+ B(0,2)*(B(1,0)*B(2,1)-B(1,1)*B(2,0));
	   	double r = detB / 2.;

			// In exact arithmetic for a symmetric matrix  -1 <= r <= 1
			// but computation error can leave it slightly outside this range.
			double phi;
   		if (r <= -1)
      		phi = M_PI / 3.;
   		else if (r >= 1)
      		phi = 0;
   		else
      		phi = acos(r) / 3.;

			// the eigenvalues satisfy eig3 <= eig2 <= eig1
   		eigs[0] = q + 2. * p * cos(phi);
   		eigs[2] = q + 2. * p * cos(phi + (2.*M_PI/3.));
   		eigs[1] = 3. * q - eigs[0] - eigs[2];     // since trace(cov) = eig1 + eig2 + eig3

			if (std::abs(eigs[0]) < 1.e-16)
				dir *= 0.;
			else {
				Matrix<double> prod(3,3);
				prod = (cov - eigs[1] * I) * (cov - eigs[2] * I);
				int ind = 0;
				double dirnorm = 0;
				do {
					dir[0] = prod(0,ind);
					dir[1] = prod(1,ind);
					dir[2] = prod(2,ind);
					dirnorm = sqrt(dir[0]*dir[0]+dir[1]*dir[1]+dir[2]*dir[2]);
					ind++;
				}
				while ((dirnorm < 1.e-10) && (ind < 3));
				assert(dirnorm >= 1.e-10);
				dir[0] /= dirnorm;
				dir[1] /= dirnorm;
				dir[2] /= dirnorm;
			}

		}


		// Construction des paquets enfants
		curr->son[0] = new Cluster(curr->depth+1);
		curr->son[1] = new Cluster(curr->depth+1);
		std::vector<int> num0;
		std::vector<int> num1;

		for(int j=0; j<nb_pt; j++){
			R3 dx = x[tab[num[j]]] - xc;
			// std::cout <<(dir,dx) << std::endl;
			if( (dir,dx)>0 ){
				num0.push_back(num[j]);
			}
			else{
				num1.push_back(num[j]);
			}
		}
		curr->son[0]->set_offset(curr->offset);
		curr->son[1]->set_offset(curr->offset+num0.size());
		curr->son[0]->set_size(num0.size());
		curr->son[1]->set_size(num1.size());

		// Recursivite
		if((num0.size() >= Parametres::minclustersize) && (num1.size() >= Parametres::minclustersize)) {
			s.push(curr->son[0]);
			s.push(curr->son[1]);
			n.push(num0);
			n.push(num1);
		}
		else{
			this->max_depth= std::max(this->max_depth,curr ->depth);
			if (this->min_depth<0) {this->min_depth=curr->depth;}
			else{
			this->min_depth= std::min(this->min_depth,curr ->depth);}

			delete curr->son[0]; curr->son[0] = NULL;
			delete curr->son[1]; curr->son[1] = NULL;

			std::copy_n(num.begin(),num.size(),perm.begin()+curr->offset);

		}
	}
}

// Full constructor
Cluster::Cluster(const std::vector<R3>& x0, const std::vector<double>& r0,const std::vector<int>& tab0, const std::vector<double>& g0, std::vector<int>& perm){
	this->build(x0,r0,tab0,g0,perm);
}

// Constructor without radius
Cluster::Cluster(const std::vector<R3>& x0, const std::vector<int>& tab0, const std::vector<double>& g0, std::vector<int>& perm){
	this->build(x0,std::vector<double>(x0.size(),0),tab0,g0,perm);
}

// Constructor without mass
Cluster::Cluster(const std::vector<R3>& x0, const std::vector<double>& r0, const std::vector<int>& tab0,std::vector<int>& perm){
	this->build(x0,r0,tab0,std::vector<double>(x0.size(),1),perm);
}

// Constructor without tab
Cluster::Cluster(const std::vector<R3>& x0, const std::vector<double>& r0, const std::vector<double>& g0, std::vector<int>& perm){
	std::vector<int> tab0(x0.size());
	std::iota(tab0.begin(),tab0.end(),int(0));
	this->build(x0,std::vector<double>(x0.size(),0),tab0,std::vector<double>(x0.size(),1),perm);
}

// Constructor without mass and rad
Cluster::Cluster(const std::vector<R3>& x0, const std::vector<int>& tab0, std::vector<int>& perm){
	this->build(x0,std::vector<double>(x0.size(),0),tab0,std::vector<double>(x0.size(),1),perm);
}

// Constructor without tab, mass and rad
Cluster::Cluster(const std::vector<R3>& x0, std::vector<int>& perm){
	std::vector<int> tab0(x0.size());
	std::iota(tab0.begin(),tab0.end(),int(0));
	this->build(x0,std::vector<double>(x0.size(),0),tab0,std::vector<double>(x0.size(),1),perm);
}


// On utilise le fait qu'on a toujours ndofperelt dofs par element geometrique
// void TraversalBuildLabel(const Cluster& t, const std::vector<int>& perm, std::vector<int>& labelVisu, const unsigned int visudep, const unsigned int cnt){
// 	if(t.depth<visudep){
// 		assert( t.son[0]!=0 ); // check if visudep is too high!
// 		TraversalBuildLabel(*(t.son[0]),perm,labelVisu,visudep,2*cnt);
// 		TraversalBuildLabel(*(t.son[1]),perm,labelVisu,visudep,2*cnt+1);
// 	}
// 	else{
// 		for(int i=t.offset; i<t.size/GetNdofPerElt(); i++)
// 		{
// 			labelVisu[ perm[GetNdofPerElt()*i]/GetNdofPerElt() ] = cnt-pow(2,visudep);
//
// 		}
// 	}
//
// }

void Cluster::print(const std::vector<int>& perm) const
{
	if ( !perm.empty() ) {
		std::cout << '[';
		for (std::vector<int>::const_iterator i = perm.begin()+offset; i != perm.begin()+offset+size; ++i)
		std::cout << *i << ',';
		std::cout << "\b]"<<std::endl;;
	}
	// std::cout << offset << " "<<size << std::endl;
	if (this->son[0]!=NULL) (*this->son[0]).print(perm);
	if (this->son[1]!=NULL) (*this->son[1]).print(perm);
}



//===============================//
//           BLOCK               //
//===============================//
class Block: public Parametres{

private:

	const Cluster* t;
	const Cluster* s;
	int Admissible;

public:
	Block(const Cluster& t0, const Cluster& s0):  t(&t0), s(&s0), Admissible(-1) {};
	Block(const Block& b): t(b.t), s(b.s), Admissible(b.Admissible) {};
	Block& operator=(const Block& b){t=b.t; s=b.s; Admissible=b.Admissible; return *this;}
	const Cluster& tgt_()const {return *(t);}
	const Cluster& src_() const {return *(s);}
	void ComputeAdmissibility() {
		// Rjasanow - Steinbach (3.15) p111 Chap Approximation of Boundary Element Matrices
		Admissible =  2*std::min((*t).get_rad(),(*s).get_rad()) < eta* (norm2((*t).get_ctr()-(*s).get_ctr())-(*t).get_rad()-(*s).get_rad() )  ;
	}
	bool IsAdmissible() const{
		assert(Admissible != -1);
		return Admissible;
	}
	// friend std::ostream& operator<<(std::ostream& os, const Block& b){
	// 	os << "src:\t" << b.src_() << std::endl; os << "tgt:\t" << b.tgt_(); return os;}

};

struct comp_block
{
    inline bool operator() (const Block* block1, const Block* block2)
    {
        if (block1->tgt_().get_offset()==block2->tgt_().get_offset()){
            return block1->src_().get_offset()<block2->src_().get_offset();
        }
        else {
            return block1->tgt_().get_offset()<block2->tgt_().get_offset();
        }
    }
};


}
#endif
