#ifndef HTOOL_CLUSTERING_DDM_CLUSTER_HPP
#define HTOOL_CLUSTERING_DDM_CLUSTER_HPP

#include "cluster.hpp"
#include "splitting.hpp"
#include <stack>

namespace htool {


template<SplittingTypes SplittingType>
class DDM_Cluster: public Cluster<DDM_Cluster<SplittingType>>{
private:
	void recursive_build(const std::vector<R3>& x, const std::vector<double>& r,const std::vector<int>& tab, const std::vector<double>& g, int nb_sons, MPI_Comm comm, std::stack<DDM_Cluster*>& s, std::stack<std::vector<int>>& n){
		
		// MPI parameters
		int rankWorld, sizeWorld;
		MPI_Comm_size(comm, &sizeWorld);
		MPI_Comm_rank(comm, &rankWorld);

		while(!s.empty()){
			DDM_Cluster* curr = s.top();
			std::vector<int> num = n.top();
			s.pop();
			n.pop();

			int curr_nb_sons = curr->depth==0 ? sizeWorld : nb_sons;

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
			curr->ctr=xc;

			// Radius and covariance matrix
			Matrix<double> cov(3,3);
			double rad = 0;
			for(int j=0; j<nb_pt; j++){
				R3 u = x[tab[num[j]]] - xc;
				rad=std::max(rad,norm2(u)+r[tab[num[j]]]);
				for(int p=0; p<3; p++){
					for(int q=0; q<3; q++){
						cov(p,q) += g[tab[num[j]]]*u[p]*u[q];
					}
				}
			}
			curr->rad=rad;

			// Direction of largest extent
			double p1 = pow(cov(0,1),2) + pow(cov(0,2),2) + pow(cov(1,2),2);
			std::vector<double> eigs(3);
			Matrix<double> I(3,3);I(0,0)=1;I(1,1)=1;I(2,2)=1;
			R3 dir;
			Matrix<double> prod(3,3);
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

			// Creating sons
			curr->sons.resize(curr_nb_sons);
			for (int p=0;p<curr_nb_sons;p++){
				curr->sons[p] = new DDM_Cluster(this,(curr->counter)*curr_nb_sons+p,curr->depth+1,this->permutation);
			}
			
			// Compute numbering
			std::vector<std::vector<int>> numbering = this->splitting(x,tab,num,curr,curr_nb_sons,dir);
			
			// Set offsets, size and rank of sons
			int count = 0;

			for (int p=0;p<curr_nb_sons;p++){
				curr->sons[p]->set_offset(curr->offset+count);
				curr->sons[p]->set_size(numbering[p].size());
				count+=numbering[p].size();

				// level of parallelization
				if (curr->depth==0){
					curr->sons[p]->set_rank(curr->sons[p]->get_counter());
					if (rankWorld==curr->sons[p]->get_counter()){
							this->local_cluster = (curr->sons[p]);
					}
					this->MasterOffset[curr->sons[p]->get_counter()] = std::pair<int,int>(curr->sons[p]->get_offset(),curr->sons[p]->get_size());
				}
				// after level of parallelization
				else {
						curr->sons[p]->set_rank(curr->rank);
				}
			} 



			// Recursivite
			bool test_minclustersize=true;
			for (int p=0;p<curr_nb_sons;p++){
				test_minclustersize= test_minclustersize && (numbering[p].size() >= Parametres::minclustersize);
			}
			if(test_minclustersize) {
				for (int p=0;p<curr_nb_sons;p++){
					s.push((curr->sons[p]));
					n.push(numbering[p]);
				}
			}
			else{
				this->max_depth= std::max(this->max_depth,curr ->depth);
				if (this->min_depth<0) {this->min_depth=curr->depth;}
				else{
				this->min_depth= std::min(this->min_depth,curr ->depth);}

				for (auto & son : curr->sons){
					delete son; son = nullptr;
				}
				curr->sons.resize(0);
				std::copy_n(num.begin(),num.size(),this->permutation->begin()+curr->offset);

			}
		}
	}
public:

	// Inherhits son constructor
	using Cluster<DDM_Cluster<SplittingType>>::Cluster;

	// build cluster tree
	// nb_sons=-1 means nb_sons = 2
	void build(const std::vector<R3>& x, const std::vector<double>& r,const std::vector<int>& tab, const std::vector<double>& g, int nb_sons = 2, MPI_Comm comm=MPI_COMM_WORLD){
		assert(tab.size()==x.size()*this->ndofperelt);
		assert(x.size()==g.size());
		assert(x.size()==r.size());
		
		// MPI parameters
		int rankWorld, sizeWorld;
		MPI_Comm_size(comm, &sizeWorld);
		MPI_Comm_rank(comm, &rankWorld);
		

		// Impossible value for nb_sons
		try{
			if (nb_sons == 0 || nb_sons==1)
				throw std::string("Impossible value for nb_sons:"+NbrToStr<int>(nb_sons));
		}
		catch(std::string const& error){
			if (rankWorld){
				std::cerr << error<< std::endl;
			}
			exit(1);
		}

		// nb_sons=-1 is automatic mode
		if (nb_sons==-1){
			nb_sons=2;
		}
		// Initialisation
		this->rad = 0;
		this->size = tab.size();
		this->rank=-1;
		this->MasterOffset.resize(sizeWorld);
		this->sons.resize(sizeWorld);
		for (auto& son : this->sons ){
			son=nullptr;
		}
		this->depth = 0; // ce constructeur est appele' juste pour la racine

		this->permutation->resize(tab.size());
		std::iota(this->permutation->begin(),this->permutation->end(),0); // perm[i]=i

		// Recursion
		std::stack<DDM_Cluster*> s;
		std::stack<std::vector<int>> n;
		s.push(this);
		n.push(*(this->permutation));

		this->recursive_build(x,r,tab,g,nb_sons,comm,s,n);

	}

	// build cluster tree from given partition
	void build(const std::vector<R3>& x, const std::vector<double>& r,const std::vector<int>& tab, const std::vector<double>& g, std::vector<int> permutation0, std::vector<std::pair<int,int>> MasterOffset0, int nb_sons = 2, MPI_Comm comm=MPI_COMM_WORLD){
		assert(tab.size()==x.size()*this->ndofperelt);
		assert(x.size()==g.size());
		assert(x.size()==r.size());
		
		// MPI parameters
		int rankWorld, sizeWorld;
		MPI_Comm_size(comm, &sizeWorld);
		MPI_Comm_rank(comm, &rankWorld);
		

		// Impossible value for nb_sons
		try{
			if (nb_sons == 0 || nb_sons==1)
				throw std::string("Impossible value for nb_sons:"+NbrToStr<int>(nb_sons));
		}
		catch(std::string const& error){
			if (rankWorld){
				std::cerr << error<< std::endl;
			}
			exit(1);
		}

		// nb_sons=-1 is automatic mode
		if (nb_sons==-1){
			nb_sons=2;
		}
		
		// Initialisation of root
		this->rad = 0;
		this->size = tab.size();
		this->rank=-1;
		this->MasterOffset=MasterOffset0;
		*(this->permutation)=permutation0;
		this->depth = 0; // ce constructeur est appele' juste pour la racine

		// Build level of depth 1 with the given partition and prepare recursion
		std::stack<DDM_Cluster*> s;
		std::stack<std::vector<int>> n;

		this->sons.resize(sizeWorld);
		for (int p=0;p<sizeWorld;p++){
			this->sons[p] = new DDM_Cluster(this,p,this->depth+1,this->permutation);
			this->sons[p]->set_offset(this->MasterOffset[p].first);
			this->sons[p]->set_size(this->MasterOffset[p].second);
			this->sons[p]->set_rank(p);
			
			if (rankWorld==this->sons[p]->get_counter()){
				this->local_cluster = this->sons[p];
			}

			s.push(this->sons[p]);
			n.push(std::vector<int>(this->permutation->begin()+this->sons[p]->get_offset(),this->permutation->begin()+this->sons[p]->get_offset()+this->sons[p]->get_size()));
		}
		

		


		// Recursion
		this->recursive_build(x,r,tab,g,nb_sons,comm,s,n);

	}
	std::vector<std::vector<int>> splitting(const std::vector<R3>& x, const std::vector<int>& tab, std::vector<int>& num, Cluster<DDM_Cluster<SplittingType>> const * const curr_cluster, int nb_sons, R3 dir);

};


// Specialization of splitting
template <>
std::vector<std::vector<int>> DDM_Cluster<SplittingTypes::GeometricSplitting>::splitting(const std::vector<R3>& x, const std::vector<int>& tab, std::vector<int>& num, Cluster<DDM_Cluster<SplittingTypes::GeometricSplitting>> const * const curr_cluster, int nb_sons, R3 dir){ return geometric_splitting(x,tab,num,curr_cluster,nb_sons,dir);}

template <>
std::vector<std::vector<int>> DDM_Cluster<SplittingTypes::RegularSplitting>::splitting(const std::vector<R3>& x, const std::vector<int>& tab, std::vector<int>& num, Cluster<DDM_Cluster<SplittingTypes::RegularSplitting>> const * const curr_cluster, int nb_sons, R3 dir){ return regular_splitting(x,tab,num,curr_cluster,nb_sons,dir);}


// Typdef with specific splitting
typedef DDM_Cluster<SplittingTypes::GeometricSplitting> GeometricClusteringDDM;
typedef DDM_Cluster<SplittingTypes::RegularSplitting>   RegularClusteringDDM;



}
#endif
