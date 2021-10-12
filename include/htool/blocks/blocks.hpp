#ifndef HTOOL_BLOCKS_BLOCKS_HPP
#define HTOOL_BLOCKS_BLOCKS_HPP

#include "../clustering/cluster.hpp"
#include "admissibility_conditions.hpp"

namespace htool {

class Block {

  protected:
    // Data member
    std::vector<std::unique_ptr<Block>> sons; // Sons
    VirtualAdmissibilityCondition *admissibility_condition;
    const VirtualCluster &t;
    const VirtualCluster &s;
    bool admissible;

    double eta;
    int mintargetdepth;
    int minsourcedepth;
    unsigned int maxblocksize;

    Block *diagonal_block;
    Block *root;

    // Before computation of blocks, first guess
    std::shared_ptr<std::vector<Block *>> tasks;
    std::shared_ptr<std::vector<Block *>> local_tasks;

    // Actual leaves after computation
    std::shared_ptr<std::vector<Block *>> local_leaves;

    // Build block tree
    // False <=> current block or its sons pushed to tasks
    // True  <=> current block not pushed to tasks
    bool build_block_tree(MPI_Comm comm = MPI_COMM_WORLD) {

        std::size_t bsize = std::size_t(this->t.get_size()) * std::size_t(this->s.get_size());

        ///////////////////// Diagonal blocks
        int rankWorld;
        MPI_Comm_rank(comm, &rankWorld);

        if (this->t.get_offset() == this->t.get_local_offset() && this->t.get_size() == this->t.get_local_size() && this->s.get_offset() == this->t.get_local_offset() && this->s.get_size() == this->t.get_local_size()) {
            this->root->diagonal_block = this;
        }

        ///////////////////// Recursion
        // Admissible
        if (this->IsAdmissible() && this->t.get_rank() >= 0 && this->t.get_depth() >= this->mintargetdepth && this->s.get_depth() >= this->minsourcedepth) {
            this->tasks->push_back(this);
            return false;
        }

        else if (this->s.IsLeaf()) {
            // Leaf
            if (t.IsLeaf()) {
                return true;
            } else {
                std::vector<bool> Blocks_not_pushed(this->t.get_nb_sons());
                for (int p = 0; p < this->t.get_nb_sons(); p++) {
                    sons.push_back(std::unique_ptr<Block>(new Block(this->admissibility_condition, this->t.get_son(p), s, this->root, this->tasks, local_tasks)));
                    Blocks_not_pushed[p] = sons[p]->build_block_tree(comm);
                }

                // All sons are non admissible and not pushed to tasks -> the current block is not pushed
                if ((bsize <= maxblocksize) && std::all_of(Blocks_not_pushed.begin(), Blocks_not_pushed.end(), [](bool i) { return i; }) && this->t.get_rank() >= 0 && this->t.get_depth() >= this->mintargetdepth && this->s.get_depth() >= this->minsourcedepth) {
                    sons.clear();
                    return true;
                }
                // Some sons have been pushed, we cannot go higher. Every other sons are also pushed so that the current block is done
                else {
                    for (int p = 0; p < this->t.get_nb_sons(); p++) {
                        if (Blocks_not_pushed[p]) {
                            tasks->push_back(sons[p].get());
                        }
                    }
                    return false;
                }
            }
        } else {
            if (t.IsLeaf()) {
                std::vector<bool> Blocks_not_pushed(this->s.get_nb_sons());
                for (int p = 0; p < this->s.get_nb_sons(); p++) {
                    sons.push_back(std::unique_ptr<Block>(new Block(this->admissibility_condition, t, this->s.get_son(p), this->root, this->tasks, local_tasks)));
                    Blocks_not_pushed[p] = sons[p]->build_block_tree(comm);
                }

                if ((bsize <= maxblocksize) && std::all_of(Blocks_not_pushed.begin(), Blocks_not_pushed.end(), [](bool i) { return i; }) && this->t.get_rank() >= 0 && this->t.get_depth() >= this->mintargetdepth && this->s.get_depth() >= this->minsourcedepth) {
                    sons.clear();
                    return true;
                } else {
                    for (int p = 0; p < this->s.get_nb_sons(); p++) {
                        if (Blocks_not_pushed[p]) {
                            tasks->push_back(sons[p].get());
                        }
                    }
                    return false;
                }
            } else {
                if (this->t.get_size() > this->s.get_size()) {
                    std::vector<bool> Blocks_not_pushed(this->t.get_nb_sons());
                    for (int p = 0; p < this->t.get_nb_sons(); p++) {
                        sons.push_back(std::unique_ptr<Block>(new Block(this->admissibility_condition, this->t.get_son(p), s, this->root, this->tasks, local_tasks)));
                        Blocks_not_pushed[p] = sons[p]->build_block_tree(comm);
                    }
                    if ((bsize <= maxblocksize) && std::all_of(Blocks_not_pushed.begin(), Blocks_not_pushed.end(), [](bool i) { return i; }) && this->t.get_rank() >= 0 && this->t.get_depth() >= this->mintargetdepth && this->s.get_depth() >= this->minsourcedepth) {
                        sons.clear();
                        return true;
                    } else {
                        for (int p = 0; p < this->t.get_nb_sons(); p++) {
                            if (Blocks_not_pushed[p]) {
                                tasks->push_back(sons[p].get());
                            }
                        }
                        return false;
                    }
                } else {
                    std::vector<bool> Blocks_not_pushed(this->s.get_nb_sons());
                    for (int p = 0; p < this->s.get_nb_sons(); p++) {
                        sons.push_back(std::unique_ptr<Block>(new Block(this->admissibility_condition, t, this->s.get_son(p), this->root, this->tasks, local_tasks)));
                        Blocks_not_pushed[p] = sons[p]->build_block_tree(comm);
                    }
                    if ((bsize <= maxblocksize) && std::all_of(Blocks_not_pushed.begin(), Blocks_not_pushed.end(), [](bool i) { return i; }) && this->t.get_rank() >= 0 && this->t.get_depth() >= this->mintargetdepth && this->s.get_depth() >= this->minsourcedepth) {
                        sons.clear();
                        return true;
                    } else {
                        for (int p = 0; p < this->s.get_nb_sons(); p++) {
                            if (Blocks_not_pushed[p]) {
                                tasks->push_back(sons[p].get());
                            }
                        }
                        return false;
                    }
                }
            }
        }
    }
    bool build_sym_block_tree(MPI_Comm comm = MPI_COMM_WORLD) {

        std::size_t bsize = std::size_t(this->t.get_size()) * std::size_t(this->s.get_size());

        ///////////////////// Diagonal blocks
        int rankWorld;
        MPI_Comm_rank(comm, &rankWorld);

        if (this->t.get_offset() == this->t.get_local_offset() && this->t.get_size() == this->t.get_local_size() && this->s.get_offset() == this->t.get_local_offset() && this->s.get_size() == this->t.get_local_size()) {
            this->root->diagonal_block = this;
        }

        ///////////////////// Recursion
        // Admissible
        if (this->IsAdmissible() && t.get_rank() >= 0 && t.get_depth() >= this->mintargetdepth && s.get_depth() >= this->minsourcedepth && ((t.get_offset() == s.get_offset() && t.get_size() == s.get_size()) || (t.get_offset() != s.get_offset() && ((t.get_offset() < s.get_offset() && s.get_offset() - t.get_offset() >= t.get_size()) || (s.get_offset() < t.get_offset() && t.get_offset() - s.get_offset() >= s.get_size()))))) {
            this->tasks->push_back(this);
            return false;
        } else if (s.IsLeaf()) {
            // Leaf
            if (t.IsLeaf()) {
                return true;
            } else {
                std::vector<bool> Blocks_not_pushed(this->t.get_nb_sons());
                for (int p = 0; p < t.get_nb_sons(); p++) {
                    sons.push_back(std::unique_ptr<Block>(new Block(this->admissibility_condition, this->t.get_son(p), s, this->root, this->tasks, local_tasks)));
                    Blocks_not_pushed[p] = sons[p]->build_sym_block_tree(comm);
                }

                // All sons are non admissible and not pushed to tasks -> the current block is not pushed
                if ((bsize <= maxblocksize) && std::all_of(Blocks_not_pushed.begin(), Blocks_not_pushed.end(), [](bool i) { return i; }) && t.get_rank() >= 0 && t.get_depth() >= this->mintargetdepth && s.get_depth() >= this->minsourcedepth && ((t.get_offset() == s.get_offset() && t.get_size() == s.get_size()) || (t.get_offset() != s.get_offset() && ((t.get_offset() < s.get_offset() && s.get_offset() - t.get_offset() >= t.get_size()) || (s.get_offset() < t.get_offset() && t.get_offset() - s.get_offset() >= s.get_size()))))) {
                    sons.clear();
                    return true;
                }
                // Some sons have been pushed, we cannot go higher. Every other sons are also pushed so that the current block is done
                else {
                    for (int p = 0; p < this->t.get_nb_sons(); p++) {
                        if (Blocks_not_pushed[p]) {
                            tasks->push_back(sons[p].get());
                        }
                    }
                    return false;
                }
            }
        } else {
            if (t.IsLeaf()) {
                std::vector<bool> Blocks_not_pushed(this->s.get_nb_sons());
                for (int p = 0; p < s.get_nb_sons(); p++) {
                    sons.push_back(std::unique_ptr<Block>(new Block(this->admissibility_condition, t, this->s.get_son(p), this->root, this->tasks, local_tasks)));
                    Blocks_not_pushed[p] = sons[p]->build_sym_block_tree(comm);
                }

                if ((bsize <= maxblocksize) && std::all_of(Blocks_not_pushed.begin(), Blocks_not_pushed.end(), [](bool i) { return i; }) && t.get_rank() >= 0 && t.get_depth() >= this->mintargetdepth && s.get_depth() >= this->minsourcedepth && ((t.get_offset() == s.get_offset() && t.get_size() == s.get_size()) || (t.get_offset() != s.get_offset() && ((t.get_offset() < s.get_offset() && s.get_offset() - t.get_offset() >= t.get_size()) || (s.get_offset() < t.get_offset() && t.get_offset() - s.get_offset() >= s.get_size()))))) {
                    sons.clear();
                    return true;
                } else {
                    for (int p = 0; p < this->s.get_nb_sons(); p++) {
                        if (Blocks_not_pushed[p]) {
                            tasks->push_back(sons[p].get());
                        }
                    }
                    return false;
                }
            } else {
                std::vector<bool> Blocks_not_pushed(t.get_nb_sons() * s.get_nb_sons());
                for (int l = 0; l < s.get_nb_sons(); l++) {
                    for (int p = 0; p < t.get_nb_sons(); p++) {
                        sons.push_back(std::unique_ptr<Block>(new Block(this->admissibility_condition, t.get_son(p), s.get_son(l), this->root, this->tasks, local_tasks)));
                        Blocks_not_pushed[p + l * t.get_nb_sons()] = sons[p + l * t.get_nb_sons()]->build_sym_block_tree(comm);
                    }
                }
                if ((bsize <= maxblocksize) && std::all_of(Blocks_not_pushed.begin(), Blocks_not_pushed.end(), [](bool i) { return i; }) && t.get_rank() >= 0 && t.get_depth() >= this->mintargetdepth && s.get_depth() >= this->minsourcedepth && ((t.get_offset() == s.get_offset() && t.get_size() == s.get_size()) || (t.get_offset() != s.get_offset() && ((t.get_offset() < s.get_offset() && s.get_offset() - t.get_offset() >= t.get_size()) || (s.get_offset() < t.get_offset() && t.get_offset() - s.get_offset() >= s.get_size()))))) {
                    sons.clear();
                    return true;
                } else {
                    for (int p = 0; p < Blocks_not_pushed.size(); p++) {
                        if (Blocks_not_pushed[p]) {
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
    Block(VirtualAdmissibilityCondition *admissibility_condition0, const VirtualCluster &t0, const VirtualCluster &s0) : admissibility_condition(admissibility_condition0), t(t0), s(s0), admissible(false), eta(10), mintargetdepth(0), minsourcedepth(0), maxblocksize(1000000), diagonal_block(nullptr), root(this), tasks(std::make_shared<std::vector<Block *>>()), local_tasks(std::make_shared<std::vector<Block *>>()) {
    }

    // Node constructor
    Block(VirtualAdmissibilityCondition *admissibility_condition0, const VirtualCluster &t0, const VirtualCluster &s0, Block *root0, std::shared_ptr<std::vector<Block *>> tasks0, std::shared_ptr<std::vector<Block *>> local_tasks0) : admissibility_condition(admissibility_condition0), t(t0), s(s0), admissible(false), eta(root0->eta), mintargetdepth(root0->mintargetdepth), minsourcedepth(root0->minsourcedepth), maxblocksize(root0->maxblocksize), diagonal_block(nullptr), root(root0), tasks(tasks0), local_tasks(local_tasks0) {

        admissible = admissibility_condition->ComputeAdmissibility(t, s, eta);
    }

    // Build
    void build(char UPLO, bool force_sym = false, MPI_Comm comm = MPI_COMM_WORLD) {
        bool not_pushed;

        // Admissibility of root
        admissible = admissibility_condition->ComputeAdmissibility(t, s, this->eta);

        // Build block tree and tasks
        if (UPLO == 'U' || UPLO == 'L' || force_sym) {
            not_pushed = this->build_sym_block_tree(comm);
        } else {
            not_pushed = this->build_block_tree(comm);
        }

        if (not_pushed) {
            tasks->push_back(this);
        }

        // Build local blocks
        int local_offset = t.get_local_offset();
        int local_size   = t.get_local_size();

        for (int b = 0; b < tasks->size(); b++) {
            if (((*tasks)[b])->get_target_cluster().get_rank() == t.get_local_cluster().get_rank()) {
                if (UPLO == 'L') {
                    if (((*((*tasks)[b])).get_source_cluster().get_offset() <= (*((*tasks)[b])).get_target_cluster().get_offset() || (*((*tasks)[b])).get_source_cluster().get_offset() >= local_offset + local_size)) {

                        local_tasks->push_back((*tasks)[b]);
                    }
                } else if (UPLO == 'U') {
                    if (((*((*tasks)[b])).get_source_cluster().get_offset() >= (*((*tasks)[b])).get_target_cluster().get_offset() || (*((*tasks)[b])).get_source_cluster().get_offset() < local_offset)) {
                        local_tasks->push_back((*tasks)[b]);
                    }
                } else {
                    local_tasks->push_back((*tasks)[b]);
                }
            }
        }

        std::sort(local_tasks->begin(), local_tasks->end(), [](Block *a, Block *b) {
            return *a < *b;
        });
    }

    void build_son(const VirtualCluster &t, const VirtualCluster &s) {
        sons.push_back(std::unique_ptr<Block>(new Block(this->admissibility_condition, t, s, this->root, this->tasks, local_tasks)));
    }
    void clear_sons() { sons.clear(); }

    // Getters
    const VirtualCluster &get_target_cluster() const { return t; }
    const VirtualCluster &get_source_cluster() const { return s; }
    const Block &get_local_diagonal_block() const {
        return *(root->diagonal_block);
    }

    std::size_t get_size() const { return std::size_t(this->t.get_size()) * std::size_t(this->s.get_size()); }
    const Block &get_son(int j) const { return *(sons[j]); }
    Block &get_son(int j) { return *(sons[j]); }

    const std::vector<Block *> &get_tasks() const {
        return *tasks;
    }
    const std::vector<Block *> &get_local_tasks() const {
        return *local_tasks;
    }

    double get_eta() const { return eta; }
    int get_mintargetdepth() const { return mintargetdepth; }
    int get_minsourcedepth() const { return minsourcedepth; }
    int get_maxblocksize() const { return maxblocksize; }

    bool IsAdmissible() const {
        return admissible;
    }

    // Setters
    void set_eta(double eta0) { eta = eta0; }
    void set_mintargetdepth(int mintargetdepth0) { mintargetdepth = mintargetdepth0; }
    void set_minsourcedepth(int minsourcedepth0) { minsourcedepth = minsourcedepth0; }
    void set_maxblocksize(unsigned int maxblocksize0) {
        if (maxblocksize0 == 0) {
            throw std::invalid_argument("[Htool error] MaxBlockSize parameter cannot be zero"); // LCOV_EXCL_LINE
        }
        maxblocksize = maxblocksize0;
    }

    // Ordering
    bool operator<(const Block &block) const {
        if (t.get_offset() == block.get_target_cluster().get_offset()) {
            return s.get_offset() < block.get_source_cluster().get_offset();
        } else {
            return t.get_offset() < block.get_target_cluster().get_offset();
        }
    }
};

} // namespace htool

#endif
