#ifndef HTOOL_BLOCKS_BLOCKS_HPP
#define HTOOL_BLOCKS_BLOCKS_HPP

#include "../clustering/cluster.hpp"
#include "admissibility_conditions.hpp"

namespace htool {


template <typename ClusterImpl, template<typename> typename AdmissibilityCondition>
class Block: public Parametres{

protected:

	// Data member
	std::vector<std::unique_ptr<Block>> sons;    // Sons
	const Cluster<ClusterImpl>& t;
	const Cluster<ClusterImpl>& s;
	bool admissible;

	Block* diagonal_block;
	Block* root;


    // Before computation of blocks, first guess
	std::shared_ptr<std::vector<Block*>> tasks;
    std::shared_ptr<std::vector<Block*>> local_tasks;

    // Actual leaves after computation
    std::shared_ptr<std::vector<Block*>> local_leaves;


    // Build block tree
	// False <=> current block or its sons pushed to tasks
	// True  <=> current block not pushed to tasks
	bool build_block_tree(MPI_Comm comm=MPI_COMM_WORLD){

        int bsize =  this->t.get_size()* this->s.get_size();

		///////////////////// Diagonal blocks
		int rankWorld;
		MPI_Comm_rank(comm, &rankWorld);

		if (this->t.get_offset()==this->t.get_local_offset() && this->t.get_size() ==  this->t.get_local_size() && this->s.get_offset()==this->t.get_local_offset() && this->s.get_size() ==  this->t.get_local_size()){
			this->root->diagonal_block=this;
		}
        

		///////////////////// Recursion
		// Admissible
        if( this->IsAdmissible() &&  this->t.get_rank()>=0 &&  this->t.get_depth()>=GetMinTargetDepth() &&  this->s.get_depth()>=GetMinSourceDepth() ){
            this->tasks->push_back(this);
            return false;
        }
		
        else if( this->s.IsLeaf() ){
			// Leaf
            if( t.IsLeaf() ){
                return true;
            }
            else{
				std::vector<bool> Blocks_not_pushed( this->t.get_nb_sons());
                for (int p=0; p < this->t.get_nb_sons();p++){
                    sons.push_back(std::unique_ptr<Block>(new Block( this->t.get_son(p),s,this->root,this->tasks,local_tasks)));
					Blocks_not_pushed[p]=sons[p]->build_block_tree(comm);
                }

				// All sons are non admissible and not pushed to tasks -> the current block is not pushed
                if ((bsize <= maxblocksize) && std::all_of(Blocks_not_pushed.begin(), Blocks_not_pushed.end(),[](bool i){return i;} ) &&  this->t.get_rank()>=0 &&  this->t.get_depth()>=GetMinTargetDepth() &&  this->s.get_depth()>=GetMinSourceDepth() ) {
                    sons.clear();
                    return true;
                }
				// Some sons have been pushed, we cannot go higher. Every other sons are also pushed so that the current block is done
                else{
                    for (int p=0; p < this->t.get_nb_sons();p++){
                        if (Blocks_not_pushed[p]) {
							tasks->push_back(sons[p].get());
						}
                    }
                    return false;
                }
            }
        }
        else{
            if( t.IsLeaf() ){
				std::vector<bool> Blocks_not_pushed( this->s.get_nb_sons());
                for (int p=0; p < this->s.get_nb_sons();p++){
                    sons.push_back(std::unique_ptr<Block>(new Block(t, this->s.get_son(p),this->root,this->tasks,local_tasks)));
					Blocks_not_pushed[p] = sons[p]->build_block_tree(comm);
                }

                if ((bsize <= maxblocksize) && std::all_of(Blocks_not_pushed.begin(), Blocks_not_pushed.end(),[](bool i){return i;} ) &&  this->t.get_rank()>=0 &&  this->t.get_depth()>=GetMinTargetDepth() &&  this->s.get_depth()>=GetMinSourceDepth()) {
                    sons.clear();
                    return true;
                }
                else{
                    for (int p=0; p < this->s.get_nb_sons();p++){
                        if (Blocks_not_pushed[p]) {
							tasks->push_back(sons[p].get());
						}
                    } 
                    return false;
                }
            }
            else{
                if ( this->t.get_size()> this->s.get_size()){
					std::vector<bool> Blocks_not_pushed( this->t.get_nb_sons());
                    for (int p=0; p < this->t.get_nb_sons();p++){
                        sons.push_back(std::unique_ptr<Block>( new Block( this->t.get_son(p),s,this->root,this->tasks,local_tasks)));
						Blocks_not_pushed[p] = sons[p]->build_block_tree(comm);
                    }
                    if ((bsize <= maxblocksize) && std::all_of(Blocks_not_pushed.begin(), Blocks_not_pushed.end(),[](bool i){return i;} ) &&  this->t.get_rank()>=0 &&  this->t.get_depth()>=GetMinTargetDepth() &&  this->s.get_depth()>=GetMinSourceDepth()) {
                        sons.clear();
                        return true;
                    }
                    else{
                        for (int p=0; p < this->t.get_nb_sons();p++){
                            if (Blocks_not_pushed[p]){
								tasks->push_back(sons[p].get());
							} 
                        } 
                        return false;
                    }
                }
                else{
					std::vector<bool> Blocks_not_pushed(this->s.get_nb_sons());
                    for (int p=0; p < this->s.get_nb_sons();p++){
                        sons.push_back(std::unique_ptr<Block>(new Block(t, this->s.get_son(p),this->root,this->tasks,local_tasks)));
						Blocks_not_pushed[p]=sons[p]->build_block_tree(comm);
                    }
                    if ((bsize <= maxblocksize) && std::all_of(Blocks_not_pushed.begin(), Blocks_not_pushed.end(),[](bool i){return i;} ) &&  this->t.get_rank()>=0 &&  this->t.get_depth()>=GetMinTargetDepth() &&  this->s.get_depth()>=GetMinSourceDepth()) {
                        sons.clear();
                        return true;
                    }
                    else{
                        for (int p=0; p < this->s.get_nb_sons();p++){
                            if (Blocks_not_pushed[p]){
								tasks->push_back(sons[p].get());
							} 
                        } 
                        return false;
                    }
                }
            }
        }
	}
    bool build_sym_block_tree(MPI_Comm comm=MPI_COMM_WORLD){
            
        int bsize = t.get_size()*s.get_size();

		///////////////////// Diagonal blocks
		int rankWorld;
		MPI_Comm_rank(comm, &rankWorld);

		if (this->t.get_offset()==this->t.get_local_offset() && this->t.get_size() ==  this->t.get_local_size() && this->s.get_offset()==this->t.get_local_offset() && this->s.get_size() ==  this->t.get_local_size()){
			this->root->diagonal_block=this;
		}

        ///////////////////// Recursion
		// Admissible
        if( this->IsAdmissible() && t.get_rank()>=0 && t.get_depth()>=GetMinTargetDepth() && s.get_depth()>=GetMinSourceDepth() &&  ( (t.get_offset()==s.get_offset() && t.get_size()==s.get_size()) || (t.get_offset()!=s.get_offset() && ( (t.get_offset()<s.get_offset() && s.get_offset()-t.get_offset() >= t.get_size()) || (s.get_offset() < t.get_offset() && t.get_offset() -s.get_offset() >= s.get_size()) )) )){
            this->tasks->push_back(this);
            return false;
        }
        else if( s.IsLeaf() ){
            // Leaf
            if( t.IsLeaf() ){
                return true;
            }
            else{
                std::vector<bool> Blocks_not_pushed( this->t.get_nb_sons());
                for (int p=0; p <t.get_nb_sons();p++){
                    sons.push_back(std::unique_ptr<Block>(new Block( this->t.get_son(p),s,this->root,this->tasks,local_tasks)));
					Blocks_not_pushed[p]=sons[p]->build_sym_block_tree(comm);
                }

                // All sons are non admissible and not pushed to tasks -> the current block is not pushed
                if ((bsize <= maxblocksize) && std::all_of(Blocks_not_pushed.begin(), Blocks_not_pushed.end(),[](bool i){return i;} ) && t.get_rank()>=0 && t.get_depth()>=GetMinTargetDepth() && s.get_depth()>=GetMinSourceDepth() &&  ( (t.get_offset()==s.get_offset() && t.get_size()==s.get_size()) || (t.get_offset()!=s.get_offset() && ( (t.get_offset()<s.get_offset() && s.get_offset()-t.get_offset() >= t.get_size()) || (s.get_offset() < t.get_offset() && t.get_offset() -s.get_offset() >= s.get_size()) )) ) ) {
                    sons.clear();
                    return true;
                }
                // Some sons have been pushed, we cannot go higher. Every other sons are also pushed so that the current block is done
                else{
                    for (int p=0; p < this->t.get_nb_sons();p++){
                        if (Blocks_not_pushed[p]) {
							tasks->push_back(sons[p].get());
						}
                    }
                    return false;
                }
            }
        }
        else{
            if( t.IsLeaf() ){
                std::vector<bool> Blocks_not_pushed( this->s.get_nb_sons());
                for (int p=0; p <s.get_nb_sons();p++){
                    sons.push_back(std::unique_ptr<Block>(new Block(t, this->s.get_son(p),this->root,this->tasks,local_tasks)));
					Blocks_not_pushed[p] = sons[p]->build_sym_block_tree(comm);
                }

                if ((bsize <= maxblocksize) && std::all_of(Blocks_not_pushed.begin(), Blocks_not_pushed.end(),[](bool i){return i;} ) && t.get_rank()>=0 && t.get_depth()>=GetMinTargetDepth() && s.get_depth()>=GetMinSourceDepth()&&   ( (t.get_offset()==s.get_offset() && t.get_size()==s.get_size()) || (t.get_offset()!=s.get_offset() && ( (t.get_offset()<s.get_offset() && s.get_offset()-t.get_offset() >= t.get_size()) || (s.get_offset() < t.get_offset() && t.get_offset() -s.get_offset() >= s.get_size()) )) )) {
                    sons.clear();
                    return true;
                }
                else{
                    for (int p=0; p < this->s.get_nb_sons();p++){
                        if (Blocks_not_pushed[p]) {
							tasks->push_back(sons[p].get());
						}
                    } 
                    return false;
                }
            }
            else{
                std::vector<bool> Blocks_not_pushed(t.get_nb_sons()*s.get_nb_sons());
                for (int l=0; l <s.get_nb_sons();l++){
                    for (int p=0; p <t.get_nb_sons();p++){
                        sons.push_back(std::unique_ptr<Block>(new Block(t.get_son(p),s.get_son(l),this->root,this->tasks,local_tasks)));
                        Blocks_not_pushed[p+l*t.get_nb_sons()] = sons[p+l*t.get_nb_sons()]->build_sym_block_tree(comm);
                    }
                }
                if ((bsize <= maxblocksize) && std::all_of(Blocks_not_pushed.begin(), Blocks_not_pushed.end(),[](bool i){return i;} ) && t.get_rank()>=0 && t.get_depth()>=GetMinTargetDepth() && s.get_depth()>=GetMinSourceDepth()&&  ( (t.get_offset()==s.get_offset() && t.get_size()==s.get_size()) || (t.get_offset()!=s.get_offset() && ( (t.get_offset()<s.get_offset() && s.get_offset()-t.get_offset() >= t.get_size()) || (s.get_offset() < t.get_offset() && t.get_offset() -s.get_offset() >= s.get_size()) )) )) {
                    sons.clear();
                    return true;
                }
                else{
                    for (int p=0; p < Blocks_not_pushed.size();p++){
                        if (Blocks_not_pushed[p]){
                            tasks->push_back(sons[p].get());
                        } 
                    } 
                    return false;
                }
            }
        }
	}


public:
	
	// Root constructor
	Block(const Cluster<ClusterImpl>& t0, const Cluster<ClusterImpl>& s0):t(t0),s(s0),admissible(false),root(this),diagonal_block(nullptr),tasks(std::make_shared<std::vector<Block*>>()),local_tasks(std::make_shared<std::vector<Block*>>()){
		admissible = AdmissibilityCondition<ClusterImpl>::ComputeAdmissibility(t,s,eta);
	}

	// Node constructor
	Block(const Cluster<ClusterImpl>& t0, const Cluster<ClusterImpl>& s0, Block* root0, std::shared_ptr<std::vector<Block*>> tasks0, std::shared_ptr<std::vector<Block*>> local_tasks0):t(t0),s(s0),admissible(false),root(root0),diagonal_block(nullptr),tasks(tasks0),local_tasks(local_tasks0){
        
		admissible = AdmissibilityCondition<ClusterImpl>::ComputeAdmissibility(t,s,eta);
	}

	// Block(const Cluster<ClusterImpl>& t0, const Cluster<ClusterImpl>& s0):  t(&t0), s(&s0), Admissible(-1) {};
	// Block(const Block& b): t(b.t), s(b.s), Admissible(b.Admissible) {};
	// Block& operator=(const Block& b){t=b.t; s=b.s; Admissible=b.Admissible; return *this;}

	// Build
	void build(bool symmetric, MPI_Comm comm=MPI_COMM_WORLD){
		bool not_pushed;
		
        // Build block tree and tasks
		if (symmetric){
            not_pushed = this->build_sym_block_tree(comm);
		}
		else {
			not_pushed = this->build_block_tree(comm);
		}

		if (not_pushed){
			tasks->push_back(this);
		}

        // Build local blocks
        int local_offset = t.get_local_offset();
        int local_size = t.get_local_size();

        for(int b=0; b<tasks->size(); b++){
            if (((*tasks)[b])->get_target_cluster().get_rank() == t.get_local_cluster().get_rank()){
                if (symmetric)
                {
                    if (((*((*tasks)[b])).get_source_cluster().get_offset()<=(*((*tasks)[b])).get_target_cluster().get_offset() || (*((*tasks)[b])).get_source_cluster().get_offset()>=local_offset+local_size)){
                        local_tasks->push_back((*tasks)[b]);
                    }
                }
                else{
                    local_tasks->push_back((*tasks)[b]);
                }
            }
        }
        
        std::sort(local_tasks->begin(),local_tasks->end(),[](Block* a, Block* b ){


            return *a<*b;
        });

	}

    void build_son(const Cluster<ClusterImpl>& t,const Cluster<ClusterImpl>& s){
        sons.push_back(std::unique_ptr<Block>(new Block(t, s,this->root,this->tasks,local_tasks)));
    }
    void clear_sons(){sons.clear();}



	// Getters
	const Cluster<ClusterImpl>& get_target_cluster()const {return t;}
	const Cluster<ClusterImpl>& get_source_cluster() const {return s;}
    const Block& get_local_diagonal_block() const{
        return *(root->diagonal_block);
    }

    int          get_size() const {return t.get_size()*s.get_size();}
    const Block& get_son(int j) const {return *(sons[j]);}
    Block&       get_son(int j)       {return *(sons[j]);}

    const std::vector<Block*>& get_tasks() const{
        return *tasks;
    }
    const std::vector<Block*>& get_local_tasks() const{
        return *local_tasks;
    }

	bool IsAdmissible() const{
		return admissible;
	}



	// Ordering
	bool operator < (const Block& block) const{
        if (t.get_offset()==block.get_target_cluster().get_offset()){
            return s.get_offset()<block.get_source_cluster().get_offset();
        }
        else {
            return t.get_offset()<block.get_target_cluster().get_offset();
        }
    }

};

// struct comp_block
// {   
//     template <typename ClusterImpl>
//     inline bool operator() (const Block<ClusterImpl,AdmissibilityCondition>* block1, const Block<ClusterImpl,AdmissibilityCondition>* block2)
//     {
//         if (block1->get_target_cluster().get_offset()==block2->get_target_cluster().get_offset()){
//             return block1->get_source_cluster().get_offset()<block2->get_source_cluster().get_offset();
//         }
//         else {
//             return block1->get_target_cluster().get_offset()<block2->get_target_cluster().get_offset();
//         }
//     }
// };

}

#endif