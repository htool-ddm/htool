#ifndef CLUSTER_HPP
#define CLUSTER_HPP

#include "matrix.hpp"
#include "parametres.hpp"
#include <iomanip>

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
	const std::vector<R3>&       x;     // Nuage complet des points
	const std::vector<double>&   r;     // Rayon de champs proche pour chaque point
	const std::vector<int>&     tab;	  // Vecteur renvoyant pour chaque dof l'indice de l'entite geometrique correspondante dans x

	std::vector<int>            num;     // Indices des dofs

	Cluster*        son[2];  // Paquets enfants
	R3                 ctr;  // Centre du paquet
	double             rad;  // Rayon du champ proche

	unsigned int depth; // profondeur du cluster dans l'arbre des paquets

	Cluster(const Cluster& ); // Pas de recopie
	Cluster & operator=(const Cluster& copy_from); // pas d'affectation


public:

	Cluster(const std::vector<R3>& x0, const std::vector<double>& r0, const std::vector<int>& tab0): x(x0), r(r0), tab(tab0), ctr(), rad(0.), depth(0) {
		son[0]=0;son[1]=0;
		depth = 0; // ce constructeur est appele' juste pour la racine
		assert(tab.size()==x.size()*ndofperelt);
		for(int j=0; j<tab.size(); j++){num.push_back(j);}
	}

	Cluster(const std::vector<R3>& x0, const std::vector<double>& r0, const std::vector<int>& tab0, const unsigned int& dep): x(x0), r(r0), tab(tab0), ctr(), rad(0.) {
		son[0]=0;son[1]=0; depth = dep;
	}

	~Cluster(){if (son[0]!=0){ delete son[0];}if (son[1]!=0){ delete son[1];}};

	void build();
	bool IsLeaf() const { if(son[0]==0){return true;} return false; }

  const std::vector<R3>&  pts_() const {return x;}
  const std::vector<int>& tab_() const {return tab;}
	const double&           rad_() const {return rad;}
	const R3&               ctr_() const {return ctr;}
	const Cluster&       		son_(const int& j) const {return *(son[j]);}
	const std::vector<int>& num_() const {return num;}
	const unsigned int&   depth_() const {return depth;}

	void print()
	{
	    std::cout << num << '\n';
	    if (this->son[0]!=0) (*this->son[0]).print();
	    if (this->son[1]!=0) (*this->son[1]).print();
	}

	friend std::ostream& operator<<(std::ostream& os, const Cluster& cl){
		for(int j=0; j<(cl.num).size(); j++){os<<cl.num[j]<< "\t";}
		os<<std::endl;
		if (!cl.IsLeaf()){
			os<<cl.son_(0);
		}
		return os;
	}


	friend void TraversalBuildLabel(const Cluster& t, vectInt& labelVisu, const unsigned int visudep, const unsigned int cnt);
	friend void VisuPartitionedMesh(const Cluster& t, std::string inputname, std::string outputname, const unsigned int visudep);


};

void Cluster::build(){
	std::stack<Cluster*> s;
	s.push(this);

	while(!s.empty()){
		Cluster* curr = s.top();
		s.pop();

		// Calcul centre du paquet
		int nb_pt = curr->num.size();
		R3 xc;
		xc.fill(0);
		for(int j=0; j<nb_pt; j++){
			xc += curr->x[curr->tab[curr->num[j]]];}
		// std::cout<<"xc "<<xc<<std::endl;
		xc = (1./double(nb_pt))*xc;
		curr->ctr = xc;

		// Calcul matrice de covariance
		Matrix<double> cov(3,3);
		curr->rad=0.;
		for(int j=0; j<nb_pt; j++){
			R3 u = curr->x[curr->tab[curr->num[j]]] - xc;
			curr->rad=std::max(curr->rad,norm(u)+curr->r[curr->tab[curr->num[j]]]);
			for(int p=0; p<3; p++){
				for(int q=0; q<3; q++){
					cov(p,q) += u[p]*u[q];
				}
			}
		}

		// Calcul direction principale
		double p1 = pow(cov(0,1),2) + pow(cov(0,2),2) + pow(cov(1,2),2);
		std::vector<double> eigs(3);
		Matrix<double> I(3,3);I(0,0)=1;I(1,1)=1;I(2,2)=1;
		R3 dir;


		if (p1 < 1e-15) {
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
			if (std::abs(eigs[0]) < 1.e-15)
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
				while ((dirnorm < 1.e-15) && (ind < 3));
				assert(dirnorm >= 1.e-15);
				dir[0] /= dirnorm;
				dir[1] /= dirnorm;
				dir[2] /= dirnorm;
			}
		}

		// Construction des paquets enfants
		curr->son[0] = new Cluster(curr->x,curr->r,curr->tab,curr->depth+1);
		curr->son[1] = new Cluster(curr->x,curr->r,curr->tab,curr->depth+1);
		for(int j=0; j<nb_pt; j++){
			R3 dx = curr->x[curr->tab[curr->num[j]]] - xc;
			// std::cout <<(dir,dx) << std::endl;
			if( (dir,dx)>0 ){
				curr->son[0]->num.push_back(curr->num[j]);
			}
			else{
				curr->son[1]->num.push_back(curr->num[j]);
			}
		}

		// Recursivite
		if((curr->son[0]->num.size() >= minclustersize) && (curr->son[1]->num.size() >= minclustersize)) {
			s.push(curr->son[0]);
			s.push(curr->son[1]);
		}
		else{
			delete curr->son[0]; curr->son[0] = 0;
			delete curr->son[1]; curr->son[1] = 0;
		}
	}
}

// On utilise le fait qu'on a toujours ndofperelt dofs par element geometrique
void TraversalBuildLabel(const Cluster& t, vectInt& labelVisu, const unsigned int visudep, const unsigned int cnt){
	if(t.depth<visudep){
		assert( t.son[0]!=0 ); // check if visudep is too high!
		TraversalBuildLabel(*(t.son[0]),labelVisu,visudep,2*cnt);
		TraversalBuildLabel(*(t.son[1]),labelVisu,visudep,2*cnt+1);
	}
	else{
		for(int i=0; i<(t.num).size()/GetNdofPerElt(); i++)
		{
			labelVisu[ t.num[GetNdofPerElt()*i]/GetNdofPerElt() ] = cnt-pow(2,visudep);

		}
	}

}

void VisuPartitionedMesh(const Cluster& t, std::string inputname, std::string outputname, const unsigned int visudep){

	assert(t.depth==0); // on peut l'appeler juste pour la racine
	std::vector<R3>  X;
	std::vector<N4>  Elt;
	std::vector<int> NbPt;
	int   num,NbElt,poubelle, NbTri, NbQuad;
	R3    Pt;

	// Ouverture fichier
	std::ifstream infile;
	infile.open(inputname.c_str());
	if(!infile.good()){
		std::cout << "LoadPoints in loading.hpp: error opening the geometry file" << std::endl;
		abort();}

	// Nombre d'elements
	infile >> NbElt;
	assert(NbElt==t.x.size()/GetNdofPerElt());
	Elt.resize(NbElt);
	NbPt.resize(NbElt);

	num=0; NbTri=0; NbQuad=0;
	// Lecture elements
	for(int e=0; e<NbElt; e++){
		infile >> poubelle;
		infile >> NbPt[e];

		if(NbPt[e]==3){NbTri++;}
		if(NbPt[e]==4){NbQuad++;}

		// Calcul centre element
		for(int j=0; j<NbPt[e]; j++){
			infile >> poubelle;
			infile >> Pt;
			Elt[e][j] = num;
			X.push_back(Pt);
			num++;
		}

		// Separateur inter-element
		if(e<NbElt-1){infile >> poubelle;}

	}
	infile.close();

	vectInt labelVisu(NbElt);
	TraversalBuildLabel(t,labelVisu,visudep,1);

	// Ecriture fichier de sortie
	std::ofstream outfile;
	outfile.open((GetOutputPath()+"/"+outputname).c_str());
	outfile << "$MeshFormat\n";
	outfile << "2.2 0 8\n";
	outfile << "$EndMeshFormat\n";
	outfile << "$Nodes\n";
	outfile << X.size() << std::endl;
	for(int j=0; j<X.size(); j++){
		outfile << j+1 << "\t" << X[j] << "\n";}
	outfile << "$EndNodes\n";
	outfile << "$Elements\n";
	outfile << NbElt << std::endl;
	for(int j=0; j<NbElt; j++){
		outfile << j  << "\t";
		if(NbPt[j]==3){outfile << 2  << "\t";}
		if(NbPt[j]==4){outfile << 3  << "\t";}
		outfile << 2  << "\t";
		outfile << 99 << "\t";
		outfile << labelVisu[j] << "\t";
		for(int k=0; k<NbPt[j]; k++){
			outfile << Elt[j][k]+1 << "\t";}
		outfile << "\n";
	}
	outfile << "$EndElements\n";


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
		Admissible =  2*std::min((*t).rad_(),(*s).rad_()) < eta* (norm((*t).ctr_()-(*s).ctr_())-(*t).rad_()-(*s).rad_() )  ;
	}
	bool IsAdmissible() const{
		assert(Admissible != -1);
		return Admissible;
	}
	friend std::ostream& operator<<(std::ostream& os, const Block& b){
		os << "src:\t" << b.src_() << std::endl; os << "tgt:\t" << b.tgt_(); return os;}

};
}
#endif
