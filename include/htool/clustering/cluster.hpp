#ifndef HTOOL_CLUSTERING_CLUSTER_HPP
#define HTOOL_CLUSTERING_CLUSTER_HPP

#include "../types/matrix.hpp"
#include "../types/point.hpp"
#include "../misc/parametres.hpp"

namespace htool {


template<typename Derived>
class Cluster: public Parametres{
protected:
    // Data member
	std::vector<Derived*> sons;    // Sons

    int rank;    // Rank for dofs of the current cluster
	int depth;   // depth of the current cluster
	int counter; // numbering of the nodes level-wise

    double rad;
    R3     ctr;

	int max_depth;
	int min_depth;
	int offset;
	int size;

	std::shared_ptr<std::vector<int>> permutation;

	Derived* local_cluster;
	Derived* root;
	std::vector<std::pair<int,int>> MasterOffset;

	// Root constructor
	Cluster():counter(0),permutation(std::make_shared<std::vector<int>>()),root(static_cast<Derived*>(this)),local_cluster(nullptr){}

	// Node constructor
	Cluster(Derived* root0, int counter0, const int& dep,std::shared_ptr<std::vector<int>> permutation0):ctr(), rad(0.),max_depth(-1),min_depth(-1), offset(0), root(root0),counter(counter0),permutation(permutation0) {
		for (auto & son : sons){
			son=0;
		}
		depth = dep;
	}

	// Destructor
    ~Cluster(){
        for (int p=0;p<sons.size();p++){
            if (sons[p]!=nullptr){ delete sons[p];sons[p]=nullptr;}
        }
    };

public:

    // build cluster tree
	void build(const std::vector<R3>& x0, const std::vector<double>& r0,const std::vector<int>& tab0, const std::vector<double>& g0, int nb_sons = -1, MPI_Comm comm=MPI_COMM_WORLD){
		static_cast<Derived*>(this)->build();
	}

   

	//// Getters for local data
	const double&   get_rad() const {return rad;}
	const R3&       get_ctr() const {return ctr;}
	const Derived&  get_son(const int& j) const {return *(sons[j]);}
	Derived&        get_son(const int& j){return *(sons[j]);}
	int get_depth() const {return depth;}
	int get_rank()const {return rank;}
	int get_offset() const {return offset;}
	int get_size() const {return size;}
	int get_nb_sons() const {return sons.size();}
	int get_counter() const {return counter;}
	const Derived& get_local_cluster( MPI_Comm comm=MPI_COMM_WORLD) const {
		int rankWorld, sizeWorld;
		MPI_Comm_size(comm, &sizeWorld);
		MPI_Comm_rank(comm, &rankWorld);

		return *(root->local_cluster);
	}

	//// Getters for global data
	int	get_max_depth() const {return root->max_depth;}
	int	get_min_depth() const {return root->min_depth;}
	const std::vector<int>& get_perm() const{return *permutation;};
	std::vector<int>::const_iterator get_perm_start() const {return permutation->begin();}

	//// Getter for MasterOffsets
	int get_local_offset() const {return root->local_cluster->get_offset();}
    int get_local_size() const {return root->local_cluster->get_local_size();}
    std::pair<int,int> get_masteroffset(int i)const {return root->MasterOffset[i];}

    // Permutations
	template<typename T>
	void cluster_to_global(const T* const in, T* const out){
		for (int i = 0; i<permutation->size();i++){
			out[(*permutation)[i]]=in[i];
		}
	}
	
	template<typename T>
	void global_to_cluster(const T* const in, T* const out){
		for (int i = 0; i<permutation->size();i++){
			out[i]=in[(*permutation)[i]];
		}
	}

    // void get_offset(std::vector<int> & J, int i) const;
	// void get_size(std::vector<int> & J, int i) const;
	// void get_ctr(std::vector<R3> & ctrs, int i) const;

	//// Setters
	void set_rank  (int rank0)   {rank = rank0;}
	void set_offset(int offset0) {offset=offset0;}
	void set_size(int size0) {size=size0;}


    bool IsLeaf() const { if(sons.size()==0){return true;} return false; }

	void print() const{
		if ( !permutation->empty() ) {
			std::cout << '[';
			for (std::vector<int>::const_iterator i = permutation->begin()+offset; i != permutation->begin()+offset+size; ++i)
			std::cout << *i << ',';
			std::cout << "\b]"<<std::endl;;
		}
		// std::cout << offset << " "<<size << std::endl;
		for (auto & son : this->sons){
			if (son!=NULL) (*son).print();
		}
	}
};

}
#endif