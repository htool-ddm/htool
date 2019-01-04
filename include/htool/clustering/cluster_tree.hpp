#ifndef HTOOL_CLUSTER_TREE_HPP
#define HTOOL_CLUSTER_TREE_HPP

#include "cluster.hpp"
#include "cluster_tree_base.hpp"
#include <mpi.h>
#include <map>
#include <memory>

namespace htool {

class Cluster_tree {
private:

  // Data
  std::shared_ptr<Cluster_tree_base> base;

  std::vector<std::pair<int,int>> MasterOffset;
  const Cluster* head;
  const Cluster* local_cluster;

  int sizeWorld;
  int rankWorld;
  MPI_Comm comm;
  mutable std::map<std::string, std::string> infos;

  // Tag ranks
  void SetRanksRec(Cluster& t, const unsigned int depth, const unsigned int cnt);
  void SetRanks(Cluster& t);


  Cluster_tree(const Cluster_tree& c):base(c.base),head(c.local_cluster),local_cluster(c.local_cluster),sizeWorld(1),rankWorld(0),comm(MPI_COMM_SELF){
      MasterOffset.push_back(std::make_pair(c.MasterOffset[c.rankWorld].first,c.MasterOffset[c.rankWorld].second));
  }

public:
    // Full Constructor
    Cluster_tree(const std::vector<R3>& x0, const std::vector<double>& r0,const std::vector<int>& tab0, const std::vector<double>& g0, MPI_Comm comm0=MPI_COMM_WORLD):comm(comm0){
        base=std::make_shared<Cluster_tree_base>(x0,r0,tab0,g0);
        head = &(base->root);
        this->build();}

    // Constructor without radius
    Cluster_tree(const std::vector<R3>& x0,const std::vector<int>& tab0, const std::vector<double>& g0, MPI_Comm comm0=MPI_COMM_WORLD):comm(comm0){
        base=std::make_shared<Cluster_tree_base>(x0,tab0,g0);
        head = &(base->root);
        this->build();}

    // Constructor without mass
    Cluster_tree(const std::vector<R3>& x0, const std::vector<double>& r0,const std::vector<int>& tab0, MPI_Comm comm0=MPI_COMM_WORLD):comm(comm0){
        base=std::make_shared<Cluster_tree_base>(x0,r0,tab0);
        head = &(base->root);
        this->build();}

    // Constructor without tab
    Cluster_tree(const std::vector<R3>& x0, const std::vector<double>& r0, const std::vector<double>& g0, MPI_Comm comm0=MPI_COMM_WORLD):comm(comm0){
        base=std::make_shared<Cluster_tree_base>(x0,r0,g0);
        head = &(base->root);
        this->build();}

    // Constructor without radius and mass
    Cluster_tree(const std::vector<R3>& x0, const std::vector<int>& tab0, MPI_Comm comm0=MPI_COMM_WORLD):comm(comm0){
        base=std::make_shared<Cluster_tree_base>(x0,tab0);
        head = &(base->root);
        this->build();}

    // Constructor without radius, mass and tab
    Cluster_tree(const std::vector<R3>& x0, MPI_Comm comm0=MPI_COMM_WORLD):comm(comm0){
        base=std::make_shared<Cluster_tree_base>(x0);
        head = &(base->root);
        this->build();}

    Cluster_tree(Cluster_tree&&)                  = default; // move constructor
    Cluster_tree& operator=(Cluster_tree&&)       = default; // move assignement operator



    // Build
    void build(){

        MPI_Comm_size(comm, &sizeWorld);
        MPI_Comm_rank(comm, &rankWorld);

        // TODO better handling of this case
        if (std::pow(2,head->get_min_depth())<sizeWorld){
          std::cout << "WARNING: too many procs for the cluster tree"<< std::endl;
          std::cout << "(min_deph,sizeworld): ("<<head->get_min_depth()<<","<<sizeWorld<<")"<< std::endl;
        }

        // Infos
        infos["max_depth"]=NbrToStr<int>(head->get_max_depth());
        infos["min_depth"]=NbrToStr<int>(head->get_min_depth());
        infos["master_depth"]=NbrToStr<int>(log2(sizeWorld));


        SetRanks(base->root);
        int max_size=MasterOffset[0].second;
        int min_size=MasterOffset[0].second;
        for (int i=0;i<MasterOffset.size();i++){
            if (MasterOffset[i].second<min_size)
                min_size=MasterOffset[i].second;
            if (MasterOffset[i].second>max_size)
                max_size=MasterOffset[i].second;
        }

        infos["master_max_size"]=NbrToStr<int>(max_size);
        infos["master_min_size"]=NbrToStr<int>(min_size);


    }

    Cluster_tree create_local_cluster_tree() const{
        return Cluster_tree(*this);
    }

    // Getters
    int get_local_offset() const {return MasterOffset[rankWorld].first;}
    int get_local_size() const {return MasterOffset[rankWorld].second;}
    std::pair<int,int> get_masteroffset(int i)const {return MasterOffset[i];}
    const std::vector<int>& get_perm() const{return base->perm;};
    std::vector<std::pair<int,int>> get_masteroffset(){return MasterOffset;}
    int get_perm(int i) const{return base->perm[i];}
    const Cluster& get_head() const {return *head;}
    const Cluster& get_local_cluster() const {return *local_cluster;}
    std::vector<int>::const_iterator get_perm_start() const {return base->perm.begin();}

    // Permutations
    template<typename T>
    void cluster_to_global(const T* const in, T* const out);
    template<typename T>
    void global_to_cluster(const T* const in, T* const out);

    // Print
    void print(){head->print(base->perm);}
    void print_size(int required_depth){head->print_size(base->perm,required_depth);}

    // Output
    std::vector<int> get_labels(int visudep) const;
    void print_infos() const;
    void save_infos(const std::string& outputname, std::ios_base::openmode mode = std::ios_base::app, const std::string& sep = " = ") const;

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
            if (rankWorld==cnt-pow(2,depth)){
                local_cluster=&t;
            }
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
  for (int i = 0; i<base->perm.size();i++){
		out[base->perm[i]]=in[i];
	}
}
template<typename T>
void Cluster_tree::global_to_cluster(const T* const in, T* const out){
  for (int i = 0; i<base->perm.size();i++){
    out[i]=in[base->perm[i]];
  }
}

std::vector<int> Cluster_tree::get_labels(int visudep) const{
  std::vector<int> labels(base->perm.size());
  std::stack<const Cluster*> s_cluster;
	std::stack<int>      s_count;
  s_cluster.push(head);
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
      for(int i=curr->get_offset(); i<curr->get_offset()+curr->get_size(); i++){
        labels[ base->perm[i]] = count;
      }
    }
  }


  return labels;
}

void Cluster_tree::print_infos() const{
    if (rankWorld==0){
        for (std::map<std::string,std::string>::const_iterator it = infos.begin() ; it != infos.end() ; ++it){
            std::cout<<it->first<<"\t"<<it->second<<std::endl;
        }
    std::cout << std::endl;
    }
}

void Cluster_tree::save_infos(const std::string& outputname, std::ios_base::openmode mode, const std::string& sep ) const{
    if (rankWorld==0){
        std::ofstream outputfile(outputname, mode);
        if (outputfile){
            for (std::map<std::string,std::string>::const_iterator it = infos.begin() ; it != infos.end() ; ++it){
                outputfile<<it->first<<sep<<it->second<<std::endl;
            }
            outputfile.close();
        }
        else{
            std::cout << "Unable to create "<<outputname<<std::endl;
        }
    }
}


} // namespace

#endif
