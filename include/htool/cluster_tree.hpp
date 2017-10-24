#ifndef CLUSTER_TREE_HPP
#define CLUSTER_TREE_HPP

#include "cluster.hpp"
#include "mpi.h"

namespace htool {

class Cluster_tree {
private:
  // Data
  std::vector<int> perm;
  std::vector<std::pair<int,int>> MasterOffset;
  Cluster root;
  int sizeWorld;
  int rankWorld;
  MPI_Comm comm;

  // Tag ranks
  void SetRanksRec(Cluster& t, const unsigned int depth, const unsigned int cnt);
  void SetRanks(Cluster& t);

public:
  // Full Constructor
  Cluster_tree(const std::vector<R3>& x0, const std::vector<double>& r0,const std::vector<int>& tab0, const std::vector<double>& g0, MPI_Comm comm0=MPI_COMM_WORLD):perm(tab0.size()),root(x0,r0,tab0,g0,perm),comm(comm0){this->build();}

  // Constructor without radius
  Cluster_tree(const std::vector<R3>& x0,const std::vector<int>& tab0, const std::vector<double>& g0, MPI_Comm comm0=MPI_COMM_WORLD):perm(tab0.size()),root(x0,tab0,g0,perm),comm(comm0){this->build();}

  // Constructor without mass
  Cluster_tree(const std::vector<R3>& x0, const std::vector<double>& r0,const std::vector<int>& tab0, MPI_Comm comm0=MPI_COMM_WORLD):perm(tab0.size()),root(x0,r0,tab0,perm),comm(comm0){this->build();}

  // Constructor without tab
  Cluster_tree(const std::vector<R3>& x0, const std::vector<double>& r0, const std::vector<double>& g0, MPI_Comm comm0=MPI_COMM_WORLD):perm(x0.size()),root(x0,r0,g0,perm),comm(comm0){this->build();}

  // Constructor without radius, mass and tab
  Cluster_tree(const std::vector<R3>& x0, MPI_Comm comm0=MPI_COMM_WORLD):perm(x0.size()),root(x0,perm),comm(comm0){this->build();}

  // Build
  void build(){
    MPI_Comm_size(comm, &sizeWorld);
    MPI_Comm_rank(comm, &rankWorld);
    SetRanks(root);
    // TODO better handling of this case
    if (std::pow(2,root.get_min_depth())<sizeWorld){
      std::cout << "WARNING : too many procs for the cluster tree"<< std::endl;
    }
  }

  // Getters
  int get_local_offset() const {return MasterOffset[rankWorld].first;}
  int get_local_size() const {return MasterOffset[rankWorld].second;}
  std::pair<int,int> get_masteroffset(int i)const {return MasterOffset[i];}
  std::vector<int> get_perm(){return perm;};
  std::vector<std::pair<int,int>> get_masteroffset(){return MasterOffset;}
  int get_perm(int i){return perm[i];}
  const Cluster& get_root() const {return root;}
  std::vector<int>::const_iterator get_perm_start() const {return perm.begin();}

  // Permutations
  template<typename T>
  void cluster_to_global(const T* const in, T* const out);
  template<typename T>
  void global_to_cluster(const T* const in, T* const out);

  // Print
  void print(){root.print(perm);}

  // Output
  std::vector<int> get_labels(int visudep) const;

};


// Rank tags
void Cluster_tree::SetRanksRec(Cluster& t, const unsigned int depth, const unsigned int cnt){
	if(t.get_depth()<depth){
		t.set_rank(-1);
		SetRanksRec(t.get_son(0), depth, 2*cnt);
		SetRanksRec(t.get_son(1), depth, 2*cnt+1);
	}
	else{
		t.set_rank(cnt-pow(2,depth));
		if (t.get_depth() == depth){
			MasterOffset[cnt-pow(2,depth)] = std::pair<int,int>(t.get_offset(),t.get_size());
		}
		if (!t.IsLeaf()){
			SetRanksRec(t.get_son(0), depth, cnt);
			SetRanksRec(t.get_son(1), depth, cnt);
		}
	}
}

void Cluster_tree::SetRanks(Cluster& t){
	int rankWorld, sizeWorld;
  MPI_Comm_size(comm, &sizeWorld);
  MPI_Comm_rank(comm, &rankWorld);
  MasterOffset.resize(sizeWorld);

	SetRanksRec(t, log2(sizeWorld), 1);
}

template<typename T>
void Cluster_tree::cluster_to_global(const T* const in, T* const out){
  for (int i = 0; i<perm.size();i++){
		out[perm[i]]=in[i];
	}
}
template<typename T>
void Cluster_tree::global_to_cluster(const T* const in, T* const out){
  for (int i = 0; i<perm.size();i++){
    out[i]=in[perm[i]];
  }
}

std::vector<int> Cluster_tree::get_labels(int visudep) const{
  std::vector<int> labels(perm.size());
  std::stack<const Cluster*> s_cluster;
	std::stack<int>      s_count;
  s_cluster.push(&root);
	s_count.push(0);
  while (!(s_cluster.empty())) {
		const Cluster* const curr = s_cluster.top();
		int count     = s_count.top();
		s_cluster.pop();
		s_count.pop();
    if(curr->get_depth()<visudep){
      assert( curr->IsLeaf()!=true ); // check if visudep is too high!
			s_cluster.push(&(curr->get_son(0)));
			s_count.push(2*count);
			s_cluster.push(&(curr->get_son(1)));
			s_count.push(2*count+1);
    }
    else{
      std::cout << curr -> get_depth() << std::endl;
      for(int i=curr->get_offset(); i<curr->get_offset()+curr->get_size(); i++){
        labels[ perm[i]] = count;
      }
    }
  }


  return labels;
}

} // namespace

#endif
