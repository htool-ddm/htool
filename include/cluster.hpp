#ifndef CLUSTER_HPP
#define CLUSTER_HPP
#include <cassert>
#include "matrix.hpp"
#include "loading.hpp"
#include "parametres.hpp"
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

namespace htool {

typedef Eigen::Matrix3d               MatR3;
typedef Eigen::EigenSolver<MatR3>     EigenSolver;
typedef EigenSolver::EigenvectorsType EigenVector;
typedef EigenSolver::EigenvalueType   EigenValue;

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
	const vectR3&     x;     // Nuage complet des points
	const vectReal&   r;     // Rayon de champs proche pour chaque point
	const vectInt&  tab;	 // Vecteur renvoyant pour chaque dof l'indice de l'entite geometrique correspondante dans x
	
	vectInt         num;     // Indices des dofs
	
	Cluster*        son[2];  // Paquets enfants
	R3              ctr;     // Centre du paquet
	Real            rad;     // Rayon du champ proche
	
	unsigned int depth; // profondeur du cluster dans l'arbre des paquets
	
	void Build_Borm();
	void Build();
	
public:
	Cluster(const vectR3& x0, const vectReal& r0, const vectInt& tab0): x(x0), r(r0), tab(tab0), ctr(0.), rad(0.) {
		son[0]=0;son[1]=0;
		depth = 0; // ce constructeur est appele' juste pour la racine
		assert(tab.size()==x.size()*ndofperelt);
		for(int j=0; j<tab.size(); j++){num.push_back(j);}
		// Nouvel algorithme qui calcule le barycentre et le rayon max pour le nuage de points du cluster :
		Build();
		
		// page 45 et algorithme 4 du livre de Borm :
		//		Build_Borm();
		//		NearFieldBall();
	}
	
	Cluster(const vectR3& x0, const vectReal& r0, const vectInt& tab0, const unsigned int& dep): x(x0), r(r0), tab(tab0), ctr(0.), rad(0.) {
		son[0]=0;son[1]=0; depth = dep;}
	Cluster(const Cluster& ); // Pas de recopie
	~Cluster(){if (son[0]!=0){ delete son[0];}if (son[1]!=0){ delete son[1];}};
	void NearFieldBall();
	bool IsLeaf() const { if(son[0]==0){return true;} return false; }
	void push_back(const int& j){num.push_back(j);}
	
	friend const vectR3&  pts_(const Cluster& t){return t.x;}
    friend const vectInt& tab_(const Cluster& t){return t.tab;}
	friend const Real&    rad_(const Cluster& t){return t.rad;}
	friend const R3&      ctr_(const Cluster& t){return t.ctr;}
	friend Cluster&       son_(const Cluster& t,const int& j){return *(t.son[j]);}
	friend const vectInt& num_(const Cluster& t){return t.num;}
	friend ostream& operator<<(ostream& os, const Cluster& cl){
		for(int j=0; j<(cl.num).size(); j++){os<<cl.num[j]<< "\t";} return os;}
	friend void DisplayTree(const Cluster&);
	
	friend void TraversalBuildLabel(const Cluster& t, vectInt& labelVisu, const unsigned int visudep, const unsigned int cnt);
	friend void VisuPartitionedMesh(const Cluster& t, std::string inputname, std::string outputname, const unsigned int visudep);
	
	
};

//void DisplayTree(const Cluster& cl){
//	cout << cl << endl;
//	if(!cl.IsLeaf()){
//		DisplayTree(son_(cl,0));
//		DisplayTree(son_(cl,1));}
//}

void Cluster::Build_Borm(){
	// Calcul centre du paquet
	int nb_dof = num.size();
	R3 xc = 0.;
	for(int j=0; j<nb_dof; j++){
		xc += x[tab[num[j]]];}
	xc = (1./Real(nb_dof))*xc;
	
	// Calcul matrice de covariance
	MatR3 cov; cov.setZero();
	for(int j=0; j<nb_dof; j++){
		R3 u = x[tab[num[j]]] - xc;
		for(int p=0; p<3; p++){
			for(int q=0; q<3; q++){
				cov(p,q) += u[p]*u[q];
			}
		}
	}
	
	// Calcul direction principale
	EigenSolver eig(cov);
	EigenValue  lambda = eig.eigenvalues();
	EigenVector ev = eig.eigenvectors();
	int l = 0; Real max=std::abs(lambda[0]);
	if( max<std::abs(lambda[1]) ){l=1; max=std::abs(lambda[1]);}
	if( max<std::abs(lambda[2]) ){l=2; }
	R3 w;
	w[0] = ev(0,l).real();
	w[1] = ev(1,l).real();
	w[2] = ev(2,l).real();
	
	// Construction des paquets enfants
	if(num.size()!= ndofperelt){
		son[0] = new Cluster(x,r,tab,depth+1);
		son[1] = new Cluster(x,r,tab,depth+1);
		for(int j=0; j<nb_dof; j++){
			R3 dx = x[tab[num[j]]] - xc;
			if( (w,dx)>0 ){
				son[0]->push_back(num[j]);
			}
			else{
				son[1]->push_back(num[j]);
			}
		}
	}
	
	// Recursivite
	if(num.size()!= ndofperelt){ // On utilise le fait qu'on a toujours ndofperelt dofs par element geometrique
		assert( son[0]!=0 );
		assert( son[1]!=0 );
		son[0]->Build_Borm();
		son[1]->Build_Borm();
	}
	
	// Feuille de l'arbre
	if(son[0]==0){
		ctr=x[tab[num[0]]];
		rad=r[tab[num[0]]];
	}
	else {
		// Centre et rayon champ proche
		const Real& r0 = son[0]->rad;
		const Real& r1 = son[1]->rad;
		const R3&   c0 = son[0]->ctr;
		const R3&   c1 = son[1]->ctr;
		
		Real l = 0.5*( 1 + (r1-r0)/norm(c1-c0) );
		ctr = (1-l)*c0 + l*c1;
		rad = l*norm(c1-c0)+r0;
	}
	
	
}

void Cluster::Build(){
	std::stack<Cluster*> s;
	s.push(this);
	
	while(!s.empty()){
		Cluster* curr = s.top();
		s.pop();	
	
		// Calcul centre du paquet
		int nb_pt = curr->num.size();
		R3 xc = 0.;
		for(int j=0; j<nb_pt; j++){
			xc += curr->x[curr->tab[curr->num[j]]];}
		xc = (1./Real(nb_pt))*xc;
	
		curr->ctr = xc;
	
		//	Real radmax=0.;
		//	for(int j=0; j<nb_pt; j++){
		//		if( radmax<norm(xc-x[tab[num[j]]]) ){
		//			radmax=norm(xc-x[tab[num[j]]]);} }
	
		// Calcul matrice de covariance
		MatR3 cov; cov.setZero();
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
		Real p1 = pow(cov(0,1),2) + pow(cov(0,2),2) + pow(cov(1,2),2);
		vectReal eigs(3);
		MatR3 I = MatR3::Identity();
		if (p1 == 0) {
	    	// cov is diagonal.
	   		eigs[0] = cov(0,0);
	   		eigs[1] = cov(1,1);
	   		eigs[2] = cov(2,2);
	   		if (eigs[0] < eigs[1]) {
	   			Real tmp = eigs[0];
	   			eigs[0] = eigs[1];
	   			eigs[1] = tmp;
	   		}
	   		if (eigs[0] < eigs[2]) {
	   			Real tmp = eigs[0];
	   			eigs[0] = eigs[2];
	   			eigs[2] = tmp;
	 		}   		
		}
		else {
			Real q = (cov(0,0)+cov(1,1)+cov(2,2))/3.;
	   		Real p2 = pow(cov(0,0) - q,2) + pow(cov(1,1) - q,2) + pow(cov(2,2) - q,2) + 2. * p1;
	   		Real p = sqrt(p2 / 6.);
	   		MatR3 B = (1. / p) * (cov - q * I);
	   		Real detB = B(0,0)*(B(1,1)*B(2,2)-B(1,2)*B(2,1))
	   					- B(0,1)*(B(1,0)*B(2,2)-B(1,2)*B(2,0))
	   					+ B(0,2)*(B(1,0)*B(2,1)-B(1,1)*B(2,0));
	   		Real r = detB / 2.;
	
			// In exact arithmetic for a symmetric matrix  -1 <= r <= 1
			// but computation error can leave it slightly outside this range.
			Real phi;
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
		}
		
		R3 dir;
		
		if (std::abs(eigs[0]) < 1.e-15)
			dir = 0;
		else {	
			MatR3 prod = (cov - eigs[1] * I) * (cov - eigs[2] * I);
			int ind = 0;
			Real dirnorm = 0;
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
		
		/*
		EigenSolver eig(cov);
		EigenValue  lambda = eig.eigenvalues();
		EigenVector ev = eig.eigenvectors();
		int l = 0; Real max=abs(lambda[0]);
		if( max<abs(lambda[1]) ){l=1; max=abs(lambda[1]);}
		if( max<abs(lambda[2]) ){l=2; }
		R3 w;
		w[0] = ev(0,l).real();
		w[1] = ev(1,l).real();
		w[2] = ev(2,l).real();
		//dir = w;
		cout << dir << " " << w << endl;
		*/
		
		// Construction des paquets enfants
		//if(curr->num.size() > 10) {
		//if(curr->num.size()!=ndofperelt){
		curr->son[0] = new Cluster(curr->x,curr->r,curr->tab,curr->depth+1);
		curr->son[1] = new Cluster(curr->x,curr->r,curr->tab,curr->depth+1);
		for(int j=0; j<nb_pt; j++){
			R3 dx = curr->x[curr->tab[curr->num[j]]] - xc;
			if( (dir,dx)>0 ){
				curr->son[0]->push_back(curr->num[j]);
			}
			else{
				curr->son[1]->push_back(curr->num[j]);
			}
		}
		//}
		
		// Recursivite
		if((curr->son[0]->num.size() >= minclustersize) && (curr->son[1]->num.size() >= minclustersize)) {
		//if(curr->num.size()!=ndofperelt){ // On utilise le fait qu'on a toujours ndofperelt dofs par element geometrique
			//assert( curr->son[0]!=0 );
			//assert( curr->son[1]!=0 );
			s.push(curr->son[0]);
			s.push(curr->son[1]);
		}
		else{
			delete curr->son[0]; curr->son[0] = 0;
			delete curr->son[1]; curr->son[1] = 0;
			curr->ctr=curr->x[curr->tab[curr->num[0]]];
			curr->rad=curr->r[curr->tab[curr->num[0]]];
		}
	}
}

// Old code
//void Cluster::NearFieldBall(){
//	// Si deja construit
//	if(rad>0){return;}
//
//	// Feuille de l'arbre
//	if(son[0]==0){
//		ctr=x[num[0]];
//		rad=r[num[0]];
////		cout<<"rayon minimal :"<<r[num[0]]<<endl;
//		return;}
//
//	// Recursivite
//	son[0]->NearFieldBall();
//	son[1]->NearFieldBall();
//
//	// Centre et rayon champ proche
//	const Real& r0 = son[0]->rad;
//	const Real& r1 = son[1]->rad;
//	const R3&   c0 = son[0]->ctr;
//	const R3&   c1 = son[1]->ctr;
//
//	Real l = 0.5*( 1 + (r1-r0)/norm(c1-c0) );
//	ctr = (1-l)*c0 + l*c1;
//	rad = l*norm(c1-c0)+r0;
//
////	cout<<ctr<<" "<<rad<<endl;
//
//	}

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
	vector<R3>  X;
	vector<N4>  Elt;
	vector<int> NbPt;
	int   num,NbElt,poubelle, NbTri, NbQuad;
	R3    Pt;
	
	// Ouverture fichier
	ifstream infile;
	infile.open(inputname.c_str());
	if(!infile.good()){
		cout << "LoadPoints in loading.hpp: error opening the geometry file" << endl;
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
	ofstream outfile;
	outfile.open((GetOutputPath()+"/"+outputname).c_str());
	outfile << "$MeshFormat\n";
	outfile << "2.2 0 8\n";
	outfile << "$EndMeshFormat\n";
	outfile << "$Nodes\n";
	outfile << X.size() << endl;
	for(int j=0; j<X.size(); j++){
		outfile << j+1 << "\t" << X[j] << "\n";}
	outfile << "$EndNodes\n";
	outfile << "$Elements\n";
	outfile << NbElt << endl;
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
	friend const Cluster& tgt_(const Block& b){return *(b.t);}
	friend const Cluster& src_(const Block& b){return *(b.s);}
	void ComputeAdmissibility() {
		// Rjasanow - Steinbach (3.15) p111 Chap Approximation of Boundary Element Matrices
		Admissible = ( 2*min(rad_(*t),rad_(*s)) < eta*( norm(ctr_(*t)-ctr_(*s))-rad_(*t)-rad_(*s) ) );
	}
	bool IsAdmissible() const{
		assert(Admissible != -1);
		return Admissible;
	}
	friend ostream& operator<<(ostream& os, const Block& b){
		os << "src:\t" << src_(b) << endl; os << "tgt:\t" << tgt_(b); return os;}
	
};
}
#endif
