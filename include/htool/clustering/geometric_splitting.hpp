#ifndef HTOOL_CLUSTERING_GEOMETRIC_SPLITTING_HPP
#define HTOOL_CLUSTERING_GEOMETRIC_SPLITTING_HPP

#include "cluster.hpp"
#include <stack>

namespace htool {

class GeometricClustering: public Cluster<GeometricClustering>{

public:

	// Inherhits son constructor
	using Cluster::Cluster;

	// build cluster tree
	// nb_sons=-1 means nb_sons = sizeworld if sizeworld!=1 and nb_sons = 2 otherwise 
	// nb_sons>0 means we check that sizeworld//nb_sons if sizeworld>1
	void build(const std::vector<R3>& x, const std::vector<double>& r,const std::vector<int>& tab, const std::vector<double>& g, int nb_sons = -1, MPI_Comm comm=MPI_COMM_WORLD){
		assert(tab.size()==x.size()*ndofperelt);
		assert(x.size()==g.size());
		assert(x.size()==r.size());
		assert(nb_sons!=0);
		
		// MPI parameters
		int rankWorld, sizeWorld;
		MPI_Comm_size(comm, &sizeWorld);
		MPI_Comm_rank(comm, &rankWorld);
		

		// Impossible value for nb_sons
		try{
			if (nb_sons == 0 || nb_sons==1)
				throw std::string("Impossible value for nb_sons");
		}
		catch(std::string const& error){
			if (rankWorld){
				std::cerr << error<< std::endl;
			}
			exit(1);
		}

		//// Check compatibility between nb_sons and sizeworld
		if (nb_sons>0 && sizeWorld!=1){
			try{
				if (!(sizeWorld % nb_sons == 0))
					throw std::pair<int,int> (sizeWorld,nb_sons);
			}
			catch(std::pair<int,int> error){
				if (rankWorld==0){
					std::cerr << "Number of MPI proccesus and number of sons in clustering are not compatible:"<< std::endl;
					std::cerr << "Number of MPI proccesus = "<<error.first<< std::endl;
					std::cerr << "Number of sons in the cluster tree = "<<error.second<< std::endl;
				}
				exit(1);
			}
			rank=-1;
			MasterOffset.resize(sizeWorld);

		}
		// We fix the number of sons without MPI
		else if (nb_sons>0 && sizeWorld==1){
			rank = 0;
			local_cluster=root;
			MasterOffset.push_back(std::pair<int,int>(0,tab.size()));
		}
		// Automatic but no parallelisation, just a sane choice
		else if (nb_sons<0 && sizeWorld==1){
			nb_sons=2;
			rank = 0;
			local_cluster=root;
			MasterOffset.push_back(std::pair<int,int>(0,tab.size()));
		}
		// Automatic with parallelisation, works well for a small number of processors...
		else{
			nb_sons=sizeWorld;
			rank=-1;
			MasterOffset.resize(nb_sons);
		}

		// Initialisation
		rad = 0;
		size = tab.size();
		sons.resize(nb_sons);
		for (auto& son : sons ){
			son=nullptr;
		}
		depth = 0; // ce constructeur est appele' juste pour la racine

		permutation->resize(tab.size());
		std::iota(permutation->begin(),permutation->end(),0); // perm[i]=i

		// Recursion
		std::stack<GeometricClustering*> s;
		std::stack<std::vector<int>> n;
		s.push(this);
		n.push(*permutation);

		while(!s.empty()){
			GeometricClustering* curr = s.top();
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
			curr->sons.resize(nb_sons);
			for (int p=0;p<nb_sons;p++){
				curr->sons[p] = new GeometricClustering(this,(curr->counter)*nb_sons+p,curr->depth+1,permutation);
			}
			
			// Compute numbering
			std::vector<std::vector<int>> numbering(nb_sons);
			// For 2 sons, we can use the center of the cluster
			if (nb_sons==2){
				for(int j=0; j<nb_pt; j++){
					R3 dx = x[tab[num[j]]] - xc;

					if( (dir,dx)>0 ){
						numbering[0].push_back(num[j]);
					}
					else{
						numbering[1].push_back(num[j]);
					}
				}

			}
			// Otherwise we have to something more
			else if (num.size()>1){
				const auto minmax  = std::minmax_element(num.begin(),num.end(),[&](int a, int b){ return (x[tab[a]] - xc,dir)<(x[tab[b]] - xc,dir)  ;});
				R3 min = x[tab[*(minmax.first)]];
				R3 max = x[tab[*(minmax.second)]];
				double length = (max-min,dir)/(double)nb_sons;
				for(int j=0; j<nb_pt; j++){
					R3 dx = x[tab[num[j]]] ;
					
					int index = ((dx-min,dir))/length;
					index = (index==nb_sons) ? index-1 : index; // for max
					numbering[index].push_back(num[j]);
				}
			}
			
			// Set offsets, size and rank of sons
			int count = 0;
			int sons_at_next_level = std::pow(nb_sons,curr->depth+1);
			for (int p=0;p<nb_sons;p++){
				curr->sons[p]->set_offset(curr->offset+count);
				curr->sons[p]->set_size(numbering[p].size());
				count+=numbering[p].size();


				if (sizeWorld>1){
					// level of parallelization
					if (sons_at_next_level==sizeWorld){

						curr->sons[p]->set_rank(curr->sons[p]->get_counter());
						if (rankWorld==curr->sons[p]->get_counter()){
							local_cluster = (curr->sons[p]);
						}
						MasterOffset[curr->sons[p]->get_counter()] = std::pair<int,int>(curr->sons[p]->get_offset(),curr->sons[p]->get_size());

					}
					// before level of parallelization
					else if (sons_at_next_level<sizeWorld){
						curr->sons[p]->set_rank(-1);
					}
					// after level of parallelization
					else {
						curr->sons[p]->set_rank(curr->rank);
					}
				}
				else{
					curr->sons[p]->set_rank(curr->rank);
				}
			} 



			// Recursivite
			bool test_minclustersize=true;
			for (int p=0;p<nb_sons;p++){
				test_minclustersize= test_minclustersize && (numbering[p].size() >= Parametres::minclustersize);
			}
			if(test_minclustersize) {
				for (int p=0;p<nb_sons;p++){
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
				std::copy_n(num.begin(),num.size(),permutation->begin()+curr->offset);

			}
		}
	}

};

}
#endif
